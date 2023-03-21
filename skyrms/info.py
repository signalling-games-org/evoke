"""
Information-theoretic analyses
"""

import numpy as np
import scipy.optimize as opt
from scipy import sparse

from skyrms.asymmetric_games import Chance

class Information:
    """
    Calculate information-theoretic quantities between strats. It expects a
    game, as created by game.Chance or game.NonChance, a sender strategy, and a
    receiver strategy
    """

    def __init__(self, game, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        self.game = game
        if isinstance(game, Chance):
            self.state_chances = game.state_chances
            self.msg_cond_on_states, self.acts_cond_on_msg = (self.sender,
                                                              self.receiver)
        else:
            # This is a game without a chance node. State chances are
            # calculated from the sender strategy, and msg_cond_on_states is
            # not given directly
            self.state_chances = np.sum(sender, axis=1)
            self.msg_cond_on_states, self.acts_cond_on_msg = (
                from_joint_to_conditional(sender), self.receiver)

    def joint_states_messages(self):
        return from_conditional_to_joint(self.state_chances,
                                         self.msg_cond_on_states)

    def joint_messages_acts(self):
        joint_s_m = self.joint_states_messages()
        uncondmessages = sum(joint_s_m)
        return from_conditional_to_joint(uncondmessages, self.acts_cond_on_msg)

    def joint_states_acts(self):
        joint_s_m = self.joint_states_messages()
        return joint_s_m.dot(self.acts_cond_on_msg)

    def mutual_info_states_acts(self):
        """
        Calculate the mutual info between states and acts
        """
        return mutual_info_from_joint(self.joint_states_acts())

    def mutual_info_states_messages(self):
        """
        Calculate the mutual info between states and messages
        """
        return mutual_info_from_joint(self.joint_states_messages())

    def mutual_info_messages_acts(self):
        """
        Calculate the mutual info between messages and acts
        """
        return mutual_info_from_joint(self.joint_messages_acts())


class RDT:
    """
    Calculate the rate-distortion function for a game and any number of distortion
    measures
    """

    def __init__(self, game, dist_tensor=None, epsilon=0.001):
        """
        Parameters
        ----------
        game: a skyrms.asymmetric_games.Chance object

        dist_tensor: a collection of distortion matrices (same dimensions as payoff
        matrices), stacked along axis 2

        epsilon: the precision up to which the point should be calculated
        """
        self.pmf = game.state_chances
        self.game = game
        self.epsilon = epsilon
        if type(dist_tensor) == np.ndarray:
            self.dist_tensor = dist_tensor
        else:
            self.dist_tensor = self.dist_tensor_from_game()
        self.outcomes = self.dist_tensor.shape[-1]

    def dist_tensor_from_game(self):
        """
        Return normalize_distortion() for sender and receiver payoffs
        """
        return np.array([
            normalize_distortion(self.game.sender_payoff_matrix),
            normalize_distortion(self.game.receiver_payoff_matrix)
        ])

    def blahut(self, lambda_, max_rounds=100, return_cond=False):
        """
        Calculate the point in the R(D, D') surface with slopes given by
        lambda_ and mu_. Follows Cover & Thomas 2006, p. 334
        """
        # we start with the uniform output distribution
        params = len(lambda_)
        output = np.ones(self.outcomes) / self.outcomes
        cond = self.update_conditional(lambda_, output)
        rate = self.calc_rate(cond, output)
        delta_dist = 2 * self.epsilon
        rounds = 0
        while delta_dist > self.epsilon and rounds <= max_rounds:
            output = self.pmf @ cond
            cond = self.update_conditional(lambda_, output)
            new_rate = self.calc_rate(cond, output)
            delta_dist = np.abs(new_rate - rate)
            rate = new_rate
            rounds = rounds + 1
            if rounds == max_rounds:
                print("Max rounds for {}".format(lambda_))
        distortion = [
            self.calc_distortion(cond, matrix) for matrix in range(params)
        ]
        if return_cond:
            return_tuple = (rate, *distortion, cond)
        else:
            return_tuple = (rate, *distortion)
        return return_tuple

    def update_conditional(self, lambda_, output):
        """
        Calculate a new conditional distribution from the <output> distribution
        and the <lambda_> parameters.  The conditional probability matrix is such that
        cond[i, j] corresponds to P(x^_j | x_i)
        """
        params = len(lambda_)
        axes = tuple([Ellipsis] + [np.newaxis] * params)
        lagrange = (-1 * lambda_[axes] * self.dist_tensor).sum(0)
        cond = output * np.exp(lagrange)
        return normalize_axis(cond, 1)

    def calc_distortion(self, cond, matrix):
        """
        Calculate the distortion for a given channel (individuated by the
        conditional matrix in <cond>), for a certain slice of self.dist_tensor
        """
        # return np.sum(self.pmf @ (cond * self.dist_matrix))
        return np.matmul(self.pmf, (cond * self.dist_tensor[matrix])).sum()

    def calc_rate(self, cond, output):
        """
        Calculate the rate for a channel (given by <cond>) and output
        distribution (given by <output>)
        """
        return np.sum(self.pmf @ (cond * np.ma.log2(cond / output).filled(0)))

    def from_cond_to_RD(self, cond, dist_measures):
        """
        Take a channel matrix, cond, where cond[i, j] gives P(q^[j] | q[i]),
        and calculate rate and distortions for it.
        <dist_measures> is a list of integers stating which distortion measures
        we want.
        """
        output = self.pmf @ cond
        rate = self.calc_rate(cond, output)
        distortion = [
            self.calc_distortion(cond, matrix) for matrix in dist_measures
        ]
        return (rate, *distortion)


class OptimizeRate(RDT):
    """
    A class to calculate rate-distortion surface with a scipy optimizer
    """

    def __init__(self,
                 game,
                 dist_measures=None,
                 dist_tensor=None,
                 epsilon=1e-4):
        """
        Parameters
        ----------

        dist_measures: A list of integers with the distortions from
        self.dist_tensor to be considered
        """
        RDT.__init__(self, game, dist_tensor, epsilon)
        self.states = len(self.pmf)
        self.acts = game.acts
        if dist_measures:
            self.dist_measures = dist_measures
        else:
            self.dist_measures = range(self.dist_tensor.shape[0])
        self.hess = opt.BFGS(exception_strategy='skip_update')
        self.bounds = opt.Bounds(0, 1)
        self.constraint = self.lin_constraint()
        self.default_cond_init = self.cond_init()

    def make_calc_RD(self):
        """
        Return a function that calculates an RD (hyper-)surface using the
        trust-constr scipy optimizer, for a given list of distortion
        objectives.
        """

        def calc_RD(distortions,
                    cond_init=self.default_cond_init,
                    return_obj=False,
                    **kwargs):
            result = opt.minimize(
                self.rate,
                cond_init,
                method="trust-constr",
                jac='2-point',
                hess=self.hess,
                constraints=[self.gen_lin_constraint(distortions)],
                bounds=self.bounds,
                **kwargs)
            if return_obj:
                return np.array([result.status, result.fun]), result
            return np.array([result.status, result.fun])

        return calc_RD

    def rate(self, cond_flat):
        """
        Calculate rate for make_calc_RD()
        """
        cond = cond_flat.reshape(self.states, self.outcomes)
        output = self.pmf @ cond
        return self.calc_rate(cond, output)

    def cond_init(self):
        """
        Return an initial conditional matrix
        """
        return np.ones((self.states * self.outcomes)) / self.outcomes

    def gen_lin_constraint(self, distortions):
        """
        Generate the LinearConstraint object

        Parameters
        ----------
        distortions: A list of distortion objectives
        """
        linear_constraint = opt.LinearConstraint(
            self.constraint, [0, 0] + [1] * self.states,
            list(distortions) + [1] * self.states)
        return linear_constraint

    def lin_constraint(self):
        """
        Collate all constraints
        """
        distortion = self.dist_constraint()
        prob = self.prob_constraint()
        return sparse.vstack((distortion, prob))

    def dist_constraint(self):
        """
        Present the distortion constraint (which is linear) the way
        scipy.optimize expects it
        """
        return np.array([
            (self.pmf[:, np.newaxis] * self.dist_tensor[measure]).flatten()
            for measure in self.dist_measures
        ])

    def prob_constraint(self):
        """
	Present the constraint that all rows in cond be probability vectors. We
        use a COO sparse matrix
	"""
        row_length = self.states * self.acts
        data = np.ones(row_length)
        rows = np.repeat(np.arange(self.states), self.states)
        columns = np.arange(row_length)
        return sparse.coo_matrix((data, (rows, columns)))


class OptimizeMessages(RDT):
    """
    A class to calculate number-of-messges/distortion curves with a scipy
    optimizer
    """

    def __init__(self,
                 game,
                 dist_measures=None,
                 dist_tensor=None,
                 epsilon=1e-4):
        """
        Parameters
        ----------

        dist_measures: A list of integers with the distortions from
        self.dist_tensor to be considered
        """
        RDT.__init__(self, game, dist_tensor, epsilon)
        self.states = len(self.pmf)
        self.acts = game.acts
        if dist_measures:
            self.dist_measures = dist_measures
        else:
            self.dist_measures = range(self.dist_tensor.shape[0])
        self.hess = opt.BFGS(exception_strategy='skip_update')
        self.bounds = opt.Bounds(0, 1)

    def make_calc_MD(self):
        """
        Return a function that calculates the minimum distortion attainable for
        a certain number of messages, using a trust-constr scipy optimizer.
        Right now it only works for one distortion measure.
        <distortion> is the distortion matrix in <dist_tensor> that we should
        care about.
        """

        def calc_MD(messages,
                    distortion,
                    codec_init_func=self.codec_init,
                    return_obj=False,
                    **kwargs):
            result = opt.minimize(
                lambda x: self.distortion(x, messages, distortion),
                codec_init_func(messages),
                method="trust-constr",
                jac='2-point',
                hess=self.hess,
                constraints=[self.gen_lin_constraint(messages)],
                bounds=self.bounds,
                **kwargs)
            if return_obj:
                return np.array([result.status, result.fun]), result
            return np.array([result.status, result.fun])

        return calc_MD

    def distortion(self, codec_flat, messages, matrix):
        """
        Calculate the distortion for a given channel (individuated by the
        conditional matrix in <cond>), for a certain slice of self.dist_tensor
        """
        coder_flat, decoder_flat = np.split(codec_flat,
                                            [self.states * messages])
        coder = coder_flat.reshape(self.states, messages)
        decoder = decoder_flat.reshape(messages, self.acts)
        cond = coder @ decoder
        return np.matmul(self.pmf, (cond * self.dist_tensor[matrix])).sum()

    def codec_init_random(self, messages):
        """
        Return an initial conditional matrix
        """
        coder_init = np.random.rand(self.states, messages)
        coder_init = (coder_init / coder_init.sum(1)[:, None]).flatten()
        decoder_init = np.random.rand(messages, self.acts)
        decoder_init = (decoder_init / decoder_init.sum(1)[:, None]).flatten()
        return np.concatenate((coder_init, decoder_init))

    def codec_init(self, messages):
        """
        Return an initial conditional matrix
        """
        coder_init = np.ones((self.states * messages)) / messages
        decoder_init = np.ones((messages * self.acts)) / self.acts
        return np.concatenate((coder_init, decoder_init))

    def gen_lin_constraint(self, messages):
        """
        Generate the LinearConstraint object

        Parameters
        ----------
        distortions: A list of distortion objectives
        """
        prob_constraint = self.prob_constraint(messages)
        linear_constraint = opt.LinearConstraint(
            prob_constraint, [1] * prob_constraint.shape[0],
            [1] * prob_constraint.shape[0])
        return linear_constraint

    def prob_constraint(self, messages):
        """
	Present the constraint that all rows in cond be probability vectors. We
        use a COO sparse matrix
	"""
        data = np.ones(messages * (self.states + self.acts))  # this is the
        # number of elements in the coder and decoder matrices
        rows_coder = np.repeat(np.arange(self.states), messages)
        columns_coder = np.arange(self.states * messages)
        rows_decoder = np.repeat(np.arange(messages), self.acts) + self.states
        columns_decoder = np.arange(
            messages * self.acts) + self.states * messages
        rows = np.concatenate((rows_coder, rows_decoder))
        columns = np.concatenate((columns_coder, columns_decoder))
        return sparse.coo_matrix((data, (rows, columns)))


class OptimizeMessageEntropy(RDT):
    """
    A class to calculate rate-distortion (where rate is actually the entropy of
    messages= with a scipy optimizer
    """

    def __init__(self,
                 game,
                 dist_measures=None,
                 dist_tensor=None,
                 messages=None,
                 epsilon=1e-4):
        """
        Parameters
        ----------

        dist_measures: A list of integers with the distortions from
        self.dist_tensor to be considered
        """
        RDT.__init__(self, game, dist_tensor, epsilon)
        self.states = len(self.pmf)
        if dist_measures:
            self.dist_measures = dist_measures
        else:
            self.dist_measures = range(self.dist_tensor.shape[0])
        if messages:
            self.messages = messages
        else:
            self.messages = self.states
        self.enc_dec_length = self.messages * (self.states + self.outcomes)
        # length of the encoder_decoder vector we optimize over.
        self.hess = opt.BFGS(exception_strategy='skip_update')
        self.bounds = opt.Bounds(0, 1)
        self.default_enc_dec_init = self.enc_dec_init()

    def make_calc_RD(self):
        """
        Return a function that calculates an RD (hyper-)surface using the
        trust-constr scipy optimizer, for a given list of distortion
        objectives.
        """

        def calc_RD(distortions,
                    enc_dec_init=self.default_enc_dec_init,
                    return_obj=False):
            result = opt.minimize(
                self.message_entropy,
                enc_dec_init,
                method="trust-constr",
                jac='2-point',
                hess=self.hess,
                constraints=([self.gen_lin_constraint()] +
                             self.gen_nonlin_constraint(distortions)),
                bounds=self.bounds)
            if return_obj:
                return np.array([result.status, result.fun]), result
            return np.array([result.status, result.fun])

        return calc_RD

    def minimize_distortion(self, matrix):
        """
        Return a function that finds an encoder-decoder pair, with the
        requisite dimension, that minimizes a single distortion objective, using a
        trust-constr scipy optimizer        
        """

        def min_dist(enc_dec_init=self.default_enc_dec_init, return_obj=False):
            result = opt.minimize(self.gen_dist_func(matrix),
                                  enc_dec_init,
                                  method="trust-constr",
                                  jac='2-point',
                                  hess=self.hess,
                                  constraints=self.gen_lin_constraint(),
                                  bounds=self.bounds)
            if return_obj:
                return np.array([result.status, result.fun]), result
            return np.array([result.status, result.fun])

        return min_dist

    def message_entropy(self, encode_decode):
        """
        Calculate message entropy given an encoder-decoder pair, where the two
        matrices are flattened and then concatenated
        """
        encoder = self.reconstruct_enc_dec(encode_decode,
                                           reconstruct_decoder=False)
        message_probs = self.pmf @ encoder
        return entropy(message_probs)

    def reconstruct_enc_dec(self, encode_decode, reconstruct_decoder=True):
        encoder_flat, decoder_flat = np.split(encode_decode,
                                              [self.states * self.messages])
        encoder = encoder_flat.reshape(self.states, self.messages)
        if reconstruct_decoder:
            decoder = decoder_flat.reshape(self.messages, self.game.acts)
            return encoder, decoder
        return encoder

    def enc_dec_init(self):
        """
        Return an initial conditional matrix
        """
        encoder = np.ones((self.states * self.messages)) / self.messages
        decoder = np.ones((self.messages * self.outcomes)) / self.outcomes
        return np.concatenate((encoder, decoder))

    def gen_nonlin_constraint(self, distortions):
        """
        Generate a list of NonLinearConstraint objects

        Parameters
        ----------
        distortions: A list of distortion objectives
        """
        nonlinear_constraints = [
            opt.NonlinearConstraint(self.gen_dist_func(matrix), 0, distortion)
            for matrix, distortion in zip(self.dist_measures, distortions)
        ]
        return nonlinear_constraints

    def gen_lin_constraint(self):
        """
        Generate the LinearConstraint object 

        Parameters
        ----------
        distortions: A list of probability objectives
        """
        prob = self.prob_constraint()
        linear_constraint = opt.LinearConstraint(
            prob, [1] * (self.states + self.messages),
            [1] * (self.states + self.messages))
        return linear_constraint

    def gen_dist_func(self, matrix):
        """
        Return the function that goes into the NonLinearConstraint objects
        """

        def dist_func(encoder_decoder):
            encoder, decoder = self.reconstruct_enc_dec(encoder_decoder)
            cond = encoder @ decoder
            dist = self.calc_distortion(cond, matrix)
            return dist

        return dist_func

    def prob_constraint(self):
        """
        Present the constraint that all rows in encoder and decoder be
        probability vectors
        """
        template_encoder = np.identity(self.states)
        template_decoder = np.identity(self.messages)
        upper_left = np.repeat(template_encoder, self.messages).reshape(
            self.states, self.states * self.messages)
        lower_right = np.repeat(template_decoder, self.outcomes).reshape(
            self.messages, self.messages * self.outcomes)
        upper_right = np.zeros_like(upper_left)
        lower_left = np.zeros_like(lower_right)
        upper = np.hstack((upper_left, upper_right))
        lower = np.hstack((lower_left, lower_right))
        return np.vstack((upper, lower))


class Shea:
    """
    Calculate functional content vectors as presented in Shea, Godfrey-Smith
    andand Cao 2017.
    """

    def __init__(self, game):
        """
        Parameters
        ----------
        game: a skyrms.asymmetric_games.Chance object
        """
        self.game = game
        self.baseline_sender, self.baseline_receiver = self.baseline_payoffs()

    def baseline_payoffs(self):
        """
        Give a vector with the payoffs for sender, and another for receiver,
        when the receiver does the best possible act for it in the absence of any communication.
        I will choose, for now, the receiver act that gives the best possible sender payoff
        (this is not decided by Shea et al.; see fn.14)
        """
        vec_expected = np.vectorize(self.expected_for_act)
        payoffs = np.apply_along_axis(vec_expected, 0,
                                      np.arange(self.game.acts)).T
        maxreceiver = np.max(payoffs[:, 1])
        sender = payoffs[:, 0][payoffs[:, 1] == maxreceiver]
        return np.max(sender), maxreceiver

    def expected_for_act(self, act):
        """
        Calculate the expected payoff for sender and receiver of doing one act,
        in the absence of communication
        """
        sender_payoffs_per_state = self.game.sender_payoff_matrix[:, act]
        receiver_payoffs_per_state = self.game.receiver_payoff_matrix[:, act]
        expected_sender = self.game.state_chances.dot(sender_payoffs_per_state)
        expected_receiver = self.game.state_chances.dot(
            receiver_payoffs_per_state)
        return expected_sender, expected_receiver

    def normal_payoffs(self):
        """
        Calculate payoffs minus the baseline
        """
        normal_sender = self.game.sender_payoff_matrix - self.baseline_sender
        normal_receiver = self.game.receiver_payoff_matrix - self.baseline_receiver
        return normal_sender, normal_receiver

    def calc_dmin(self):
        """
        Calculate dmin as defined in Shea et al. (2917, p. 24)
        """
        normal_sender, normal_receiver = self.normal_payoffs()
        return np.minimum(normal_sender, normal_receiver)

    def calc_summation(self, norm_payoff, receiver_strat):
        """
        Calculate the summation in the entries of the functional content vector
        """
        inside_summation_raw = np.multiply.outer(norm_payoff, receiver_strat)
        inside_summation = np.einsum('ijkj->ijk', inside_summation_raw)
        return np.sum(inside_summation, axis=1)

    def calc_entries(self, sender_strat, receiver_strat, payoff_matrix):
        """
        Calculate the entries of the functional vector, given one choice for
        the (baselined) payoff matrix
        """
        bayes_sender = bayes_theorem(self.game.state_chances, sender_strat)
        summation = self.calc_summation(payoff_matrix, receiver_strat)
        return bayes_sender * summation

    def calc_entries_dmin(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, given one choice for
        the official dmin
        """
        return self.calc_entries(sender_strat, receiver_strat,
                                 self.calc_dmin())

    def calc_entries_sender(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, for the baselined
        sender
        """
        normal_sender = self.game.sender_payoff_matrix - self.baseline_sender
        return self.calc_entries(sender_strat, receiver_strat, normal_sender)

    def calc_entries_receiver(self, sender_strat, receiver_strat):
        """
        Calculate the entries of the functional vector, for the baselined
        receiver
        """
        normal_receiver = self.game.receiver_payoff_matrix - self.baseline_receiver
        return self.calc_entries(sender_strat, receiver_strat, normal_receiver)

    def calc_condition(self, receiver_strat, payoff_matrix, baseline):
        """
        Calculate the condition for nonzero vector entries
        """
        outer = np.multiply.outer(receiver_strat, payoff_matrix)
        precondition = np.einsum('kjij->ijk', outer)
        return np.sum(precondition, axis=1) > baseline

    def calc_condition_sender(self, receiver_strat):
        """
        Calculate condition() for the sender payoff matrix and baseline
        """
        return self.calc_condition(receiver_strat,
                                   self.game.sender_payoff_matrix,
                                   self.baseline_sender)

    def calc_condition_receiver(self, receiver_strat):
        """
        Calculate condition() for the receiver payoff matrix and baseline
        """
        return self.calc_condition(receiver_strat,
                                   self.game.receiver_payoff_matrix,
                                   self.baseline_receiver)

    def calc_condition_common(self, receiver_strat):
        """
        Calculate the condition for a nonzero functional vector entry in
        the definition in (op. cit., p. 24)
        """
        return self.calc_condition_sender(
            receiver_strat) & self.calc_condition_receiver(receiver_strat)

    def functional_content(self, entries, condition):
        """
        Put everything together in a functional vector per message
        """
        return normalize_axis(entries * condition, 0)

    def functional_content_sender(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of the sender
        """
        return self.functional_content(
            self.calc_entries_sender(sender_strat, receiver_strat),
            self.calc_condition_sender(receiver_strat))

    def functional_content_receiver(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of the receiver
        """
        return self.functional_content(
            self.calc_entries_receiver(sender_strat, receiver_strat),
            self.calc_condition_receiver(receiver_strat))

    def functional_content_dmin(self, sender_strat, receiver_strat):
        """
        Calculate the functional content from the perspective of dmin
        """
        return self.functional_content(
            self.calc_entries_dmin(sender_strat, receiver_strat),
            self.calc_condition_common(receiver_strat))


def conditional_entropy(conds, unconds):
    """
    Take a matrix of probabilities of the column random variable (r. v.)
    conditional on the row r.v.; and a vector of unconditional
    probabilities of the row r. v.. Calculate the conditional entropy of
    column r. v. on row r. v. That is:
    Input:
        [[P(B1|A1), ...., P(Bn|A1)],..., [P(B1|Am),...,P(Bn|Am)]]
        [P(A1), ..., P(Am)]
    Output:
        H(B|A)
    """
    return unconds.dot(np.apply_along_axis(entropy, 1, conds))


def mutual_info_from_joint(matrix):
    """
    Take a matrix of joint probabilities and calculate the mutual information
    between the row and column random variables
    """
    row_unconditionals = sum(matrix.transpose())
    column_unconditionals = sum(matrix)
    conditionals = from_joint_to_conditional(matrix)
    uncond_entropy = entropy(column_unconditionals)
    cond_entropy = conditional_entropy(conditionals, row_unconditionals)
    return uncond_entropy - cond_entropy


def unconditional_probabilities(unconditional_input, strategy):
    """
    Calculate the unconditional probability of messages for sender, or
    signals for receiver, given the unconditional probability of states
    (for sender) or of messages (for receiver)
    """
    return strategy * unconditional_input


def normalize_axis(array, axis):
    """
    Normalize a matrix along <axis>, being sensible with all-zero rows
    """
    return np.apply_along_axis(normalize_vector, axis, array)


def from_joint_to_conditional(array):
    """
    Normalize row-wise

    """
    return normalize_axis(array, 1)


def from_conditional_to_joint(unconds, conds):
    """
    Take a matrix of conditional probabilities of the column random variable on
    the row r. v., and a vector of unconditional probabilities of the row r. v.
    and output a matrix of joint probabilities.

    Input:
            [[P(B1|A1), ...., P(Bn|A1)],..., [P(B1|Am),...,P(Bn|Am)]]
            [P(A1), ..., P(Am)]
    Output:
            [[P(B1,A1), ...., P(Bn,A1)],..., [P(B1,Am),...,P(Bn,Am)]]
    """
    return conds * unconds[..., None]


def bayes_theorem(unconds, conds):
    """
    Perform Bayes' theorem on a matrix of conditional probabilities

    Parameters
    ----------
    unconds:
        a (n x 1) numpy array of unconditional probabilities [P(A1), ... , P(An)]
    conds:
        a (m x n) numpy array of conditional probabilities
        [[P(B1|A1), ... , P(Bm|A1)], ... , [P(B1|An), ..., P(Bm|An)]]

    Returns
    -------
    A (n x m) numpy array of conditional probabilities
        [[P(A1|B1), ... , P(An|B1)], ... , [P(PA1|Bm), ... , P(An|Bm)]]
    """
    joint = from_conditional_to_joint(unconds, conds)
    return from_joint_to_conditional(joint.T).T


def entropy(vector):
    """
    Calculate the entropy of a vector
    """
    nonzero = vector[vector > 1e-10]
    return -nonzero.dot(np.log2(nonzero))


def escalar_product_map(matrix, vector):
    """
    Take a matrix and a vector and return a matrix consisting of each element
    of the vector multiplied by the corresponding row of the matrix
    """
    trm = matrix.transpose()
    return np.vectorize(np.dot)(trm, vector).transpose()


def normalize_vector(vector):
    """
    Normalize a vector, converting all-zero vectors to uniform ones
    """
    if np.allclose(vector, np.zeros_like(vector)):
        return np.ones_like(vector) / len(vector)
    return vector / sum(vector)


def normalize_distortion(matrix):
    """
    Normalize linearly so that max corresponds to 0 distortion, and min to 1 distortion
    It must be a matrix of floats!
    """
    maxmatrix = np.max(matrix)
    minmatrix = np.min(matrix)
    numerator = maxmatrix - matrix
    denominator = maxmatrix - minmatrix
    return np.divide(numerator,
                     denominator,
                     out=np.zeros_like(matrix),
                     where=denominator != 0)
