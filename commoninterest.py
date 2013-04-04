#!/usr/bin/python3

import io, itertools, json, math, random, subprocess, sys, time

import scipy.stats

class Game:
    def __init__(self, payoffs):
        self.dimension = 3 # The dimension of the (square) game -- this is
        # hardcoded here and there; I need to change that.
        self.chances = [1/3, 1/3, 1/3]
        self.payoffs = payoffs
        self.sender, self.receiver = fromlisttomatrix(self.payoffs)
        self.kendalldistance = round(self. aggregate_kendall_distance(),2)
        self.kendallsender, self.kendallreceiver = self.intrakendall()
#        print("Act deviation: {}".format(self.actdev))
#        print("Modified Kendall tau distance: {}".format(self.kendallmoddistance))
#        print("Modified Kendall tau distance: {}".format(self.kendallmod))

    def same_best(self):
        bestactsforsender = [setofindexes(acts, max(acts)) for acts in
                self.sender]
        bestactsforreceiver = [setofindexes(acts, max(acts)) for acts in
                self.receiver]
        samebest = [sender & receiver for sender, receiver in
                zip(bestactsforsender, bestactsforreceiver)]
        return samebest


    def intrakendall(self):
        def points(state1, state2, element1, element2):
            pairwise = math.floor(abs(preferable(state1, element1, element2) -
            preferable(state2, element1, element2)))
            return pairwise 
        def kendall(state1, state2):
            return sum([points(state1, state2, pair[0], pair[1]) for pair in 
                itertools.combinations(range(self.dimension), 2)])
        skendalls = [kendall(self.sender[pair[0]], self.sender[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        rkendalls = [kendall(self.receiver[pair[0]], self.receiver[pair[1]]) for
            pair in itertools.combinations(range(self.dimension), 2)]
        return sum(skendalls)/len(skendalls), sum(rkendalls)/len(rkendalls)

    def aggregate_kendall_distance(self):
        return sum([self.chances[state] * self.kendall_tau_distance(state) for state in
            range(self.dimension)])

    def kendall_tau_distance(self, state):
        def points(sender, receiver, element1, element2):
            pairwise = math.floor(abs(preferable(sender, element1, element2) -
            preferable(receiver, element1, element2)))
            if sender[element1] == max(sender) or sender[element2] == max(sender):
                weight = 1
            elif sender[element1] == min(sender) or sender[element2] == min(sender):
                weight = 1
            return pairwise * weight
        kendall =  sum([points(self.sender[state], self.receiver[state],
            pair[0], pair[1]) for pair in
            itertools.combinations(range(self.dimension), 2)])
        return kendall

    def info_in_equilibria(self):
        gambitgame = bytes(self.write_efg(), "utf-8")
        calc_eqs = subprocess.Popen(['gambit-lcp', '-d', '3'], stdin = subprocess.PIPE,
                stdout = subprocess.PIPE)
        result = calc_eqs.communicate(input = gambitgame)[0]
        equilibria = str(result, "utf-8").split("\n")[:-1]
        sinfos, rinfos, jinfos = self.calculate_info_content(equilibria)
        return equilibria, max(sinfos), max(rinfos), max(jinfos)
        
    def write_efg(self): # This writes the game in the form Gambit
        # expects. 'output' is a file object.
        chance = self.chances
        V = self.payoffs
        filelist = []
        filelist.append(r'EFG 2 R "Untitled Extensive Game" { "Player 1" "Player 2" }')
        filelist.append("\n")
        filelist.append(r'""')
        filelist.append("\n")
        filelist.append('')
        filelist.append("\n")
        filelist.append(r'c "" 1 "" {{ "1" {0} "2" {1} "3" {2} }} 0'.format(chance[0], chance[1], chance[2]))
        filelist.append("\n")
        filelist.append(r'p "" 1 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 1 "" {{ {} , {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 2 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 3 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 4 "" {{ {}, {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 5 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 6 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 7 "" {{ {}, {} }}'.format(V[0],V[1]))
        filelist.append("\n")
        filelist.append(r't "" 8 "" {{ {}, {} }}'.format(V[2], V[3]))
        filelist.append("\n")
        filelist.append(r't "" 9 "" {{ {}, {} }}'.format(V[4], V[5]))
        filelist.append("\n")
        filelist.append(r'p "" 1 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 10 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 11 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 12 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 13 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 14 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 15 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 16 "" {{ {}, {} }}'.format(V[6], V[7]))
        filelist.append("\n")
        filelist.append(r't "" 17 "" {{ {}, {} }}'.format(V[8], V[9]))
        filelist.append("\n")
        filelist.append(r't "" 18 "" {{ {}, {} }}'.format(V[10], V[11]))
        filelist.append("\n")
        filelist.append(r'p "" 1 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r'p "" 2 1 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 19 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 20 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 21 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'p "" 2 2 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 22 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 23 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 24 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'p "" 2 3 "" { "1" "2" "3" } 0')
        filelist.append("\n")
        filelist.append(r't "" 25 "" {{ {}, {} }}'.format(V[12], V[13]))
        filelist.append("\n")
        filelist.append(r't "" 26 "" {{ {}, {} }}'.format(V[14], V[15]))
        filelist.append("\n")
        filelist.append(r't "" 27 "" {{ {}, {} }}'.format(V[16], V[17]))
        filelist.append("\n")
        filelist.append(r'</efgfile>')
        filelist.append("\n")
        stringinput = ''.join(filelist)
        return stringinput

    def calculate_Nash_eqs(self, inputfile, outputfile): # calls Gambit and
        #stores the resulting equilibria
        proc = subprocess.Popen(["gambit-lcp"], stdin = inputfile, stdout =
                outputfile)
        return proc

    def calculate_info_content(self, equilibria): # Given Gambit results, calculate in which equilibria do signals carry information
        chance = self.chances
        sinfos = []
        rinfos = []
        jinfos = []
        #print(equilibria)
        for line in equilibria:
            #print("Equilibrium", line, end =":\n")
            # The following takes a line such as "NE, 0, 0, 1, 0, 0, 1..." to a list [0, 0, 1, 0, 0, 1...]
            equilibrium = list(map(eval, line.split(sep =",")[1:]))
            mutualinfoSM, mutualinfoAM, mutualinfoSA = self.conditional_probabilities(equilibrium)
            #print(mutualinfoSA)
            sinfos.append(mutualinfoSM)
            rinfos.append(mutualinfoAM)
            jinfos.append(mutualinfoSA)
        return sinfos, rinfos, jinfos

    def conditional_probabilities(self, equilibrium): # Calculates the
        #conditional probabilities of states on signals, acts on signals, and
        #states of acts
        # Note the resulting matrices have the form: [[P(S1|M1), P(S2|M1), P(S3|M1)], [P(S1|M2), P(S2|M2), P(S3|M2)]...]
        chance = self.chances
        equilibriumsender = equilibrium[:9]
        equilibriumreceiver = equilibrium[9:]

        ### First, the information that messages carry about states ###

        conditional_probability_matrixsender = []
        unconditionalsmessages = []
        kullbackleibler = []
        for message in range(self.dimension):
            unconditional = sum([chance[i] * equilibriumsender[self.dimension * i + message] for i in range(self.dimension)]) # The unconditional probability of message
            unconditionalsmessages.append(unconditional)
            statesconditionalonmsg = []
            for state in range(self.dimension):
                conditional = chance[state] * safe_div(
                        equilibriumsender[self.dimension * state + message] , unconditional)
                statesconditionalonmsg.append(conditional)

            kld = sum([safe_kld_coefficient(conditional, unconditional) for
                conditional, unconditional in zip(statesconditionalonmsg, chance)])
            kullbackleibler.append(kld)
            #print("KL", kullbackleibler)
            conditional_probability_matrixsender.append(statesconditionalonmsg)
        averagekldsender = sum([prob * kbd for prob, kbd in zip(kullbackleibler,
            unconditionalsmessages)])

        jointprobSM = [[conditional_probability_matrixsender[message][state] *
                unconditionalsmessages[message] for state in
                range(self.dimension)] for message in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobSM",jointprobSM)

        mutualinfoSM = sum([jointprobSM[message][state] *
                safe_log(jointprobSM[message][state], self.chances[state] *
                    unconditionalsmessages[message]) for state in
                range(self.dimension) for message in range(self.dimension)])

        #print("MutualInfo SM", mutualinfoSM)

        #print('Average KL distance: {}'.format(averagekldsender))
        #print("Uncondmessages", unconditionalsmessages)

        ### Then, the information that messages carry about acts ###

        #print("eq sender {}".format(equilibriumsender))
        #print("eq receiver {}".format(equilibriumreceiver))
        conditional_probability_matrixreceiver= []
        unconditionalsacts = []
        kullbackleibler = []
        # We first calculate the unconditional probability of acts
        for act in range(self.dimension):
            unconditional = sum([unconditionalsmessages[i] *
                equilibriumreceiver[self.dimension * i + act] for i in
                range(self.dimension)]) 
            unconditionalsacts.append(unconditional)
        # Then their probability conditional on a message
        for message in range(self.dimension):
            conditionals4act = []
            if unconditionalsmessages[message] != 0:
                for act in range(self.dimension):
                    conditional = unconditionalsmessages[message] * equilibriumreceiver[self.dimension * message + act] / unconditionalsmessages[message]
                    conditionals4act.append(conditional)
                    #print("act: {}, message: {}, conditional: {}".format(act,
                        #message, conditional))
            else:
                conditionals4act=[0,0,0]
            #print("Uncondacts", unconditionalsacts)
            #print("Cond4acts", conditional)

            kld = sum([safe_kld_coefficient(conditional, unconditional) for
                conditional, unconditional in zip(conditionals4act,
                    unconditionalsacts)])
            kullbackleibler.append(kld)
            #print("KLD: {}".format(kullbackleibler))
            conditional_probability_matrixreceiver.append(conditionals4act)
        averagekldreceiver = sum([prob * kld for prob, kld in zip(
            unconditionalsmessages, kullbackleibler)])

        jointprobAM = [[conditional_probability_matrixreceiver[message][act] *
                unconditionalsmessages[message] for act in
                range(self.dimension)] for message in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobAM",jointprobAM)

        mutualinfoAM = sum([jointprobAM[message][act] *
                safe_log(jointprobAM[message][act], unconditionalsacts[act] *
                    unconditionalsmessages[message]) for act in
                range(self.dimension) for message in range(self.dimension)])

        #print("MutualInfo AM", mutualinfoAM)

        ### Finally, the info that acts carry about states

        stateconditionalonact = [[safe_div(sum([equilibriumsender[3 * state + message] *
                equilibriumreceiver[3 * message + act] *
                self.chances[state] for message in
                    range(self.dimension)]) , unconditionalsacts[act])
                        for state in range(self.dimension)] for act in
                            range(self.dimension)]

        #print("conditional prob:", stateconditionalonact)
                            
        avgkldjoint = sum([sum([safe_kld_coefficient(stateconditionalonact[act][state], 
            self.chances[state]) for state in
                range(self.dimension)]) * unconditionalsacts[act] for act in
                    range(self.dimension)])

        jointprobSA = [[stateconditionalonact[act][state] *
                unconditionalsacts[act] for state in
                range(self.dimension)] for act in range(self.dimension)]

        #print("eqbsender", equilibriumsender)
        #print("eqbreceiver", equilibriumreceiver)

        #print("jointprobSA",jointprobSA)
        #print("unconditionalsacts", unconditionalsacts)
        #print("chances", self.chances)

        mutualinfoSA = sum([jointprobSA[act][state] *
            safe_log(jointprobSA[act][state], unconditionalsacts[act] *
                self.chances[state]) for act in range(self.dimension) for state in range(self.dimension)])

        #print("MutualInfo SA", mutualinfoSA)


        return(mutualinfoSM, mutualinfoAM, mutualinfoSA)

def safe_kld_coefficient(conditional, unconditional):
                if conditional == 0 or unconditional == 0:
                    return 0
                else:
                    return  conditional * math.log2(conditional/unconditional)

def safe_log(a, b):
    try:
        return math.log2(safe_div(a,b))
    except ValueError:
        return 0

def safe_div(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return 0

def entropy(unconditional_probs):
    return sum([element * math.log2(1/element) for element in
        unconditional_probs])

#def conditional_entropy(unconditional_probs, conditional_probs):
    #return -1 * sum([ unconditional_probs[unconditional] *
        #sum(conditional_probs[conditional][unconditional] *
            #math.log2(1/conditional_probs[conditional][unconditional])))

def order_indexes(preferences):
    return [i[0] for i in sorted(enumerate(preferences), key=lambda x:x[1])]

def avg(alist):
    return round(sum(alist)/len(alist),2)

def avg_abs_dev(alist):
    return sum([abs(element - avg(alist)) for element in alist])/len(alist)

def payoffs(): # The payoff matrix, as a list
    return [random.randrange(0,100) for x in range(18)]

def fromlisttomatrix(payoff): # Takes a list of intertwined sender and receiver
    # payoffs (what payoffs() outputs) and outputs two lists of lists.
    sender = [payoff[i] for i in range(0,18,2)]
    sendermatrix = [sender[0:3],sender[3:6],sender[6:9]]
    receiver = [payoff[i] for i in range(1,18,2)]
    receivermatrix = [receiver[0:3],receiver[3:6],receiver[6:9]]
    return sendermatrix, receivermatrix

def preferable(ranking, element1, element2): # returns 0 if element1 is
    # preferable; 0.5 if both equally preferable; 1 if element2 is preferable
    index1 = ranking[element1]
    index2 = ranking[element2]
    if index2 > index1:
        return 0
    if index2 == index1:
        return 0.5
    if index2 < index1:
        return 1

def setofindexes(originallist, element):
    return set([i for i in range(len(originallist)) if originallist[i] ==
        element]) # We return sets -- later we are doing intersections

def normalize_matrix(matrix):
    flatmatrix = [i for i in itertools.chain.from_iterable(matrix)] # what's
    # the right way to do this?
    bottom = min(flatmatrix)
    top = max(flatmatrix)
    return [[(element - bottom)/(top - bottom) for element in row] for row in matrix]

def order_list(alist):
    olist = sorted(alist, reverse=True)
    return [olist.index(element) for element in alist]

def main():
    games = {}
    timestr = time.strftime("%d%b%H-%M")
    counter = 0
    while counter < 10:
        #print(counter)
        #print()
        entry = {}
        game = Game(payoffs())
        entry["kendall"] = game.kendalldistance
        if 2 < game.kendalldistance: 
        #while game.payofftype != [0,3,0]: 
        #    game = Game(payoffs())
            game.maxinfo = game.info_in_equilibria() 
            if game.maxinfo[1] > 0:
                entry["sender"] = game.sender
                entry["receiver"] = game.receiver
                entry["payoffs"] = game.payoffs
                entry["maxinfo"] = game.maxinfo
                games[counter] = entry
                counter += 1
        #print()
#print(games)
    gamesname = ''.join(["nugget", timestr])
    with open(gamesname, 'w') as gamefile:
        json.dump(games, gamefile)

def main2():
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        for payofftype in types:
            #print(payofftype)
            typecode = payofftype[0]*100 + payofftype[1]*10 + payofftype[2]
            for i in range(40):
                print("EXPERIMENT", i)
                #print()
                entry = {}
                game = Game(payoffs())
                while game.payofftype != payofftype: 
                    game = Game(payoffs())
                datapoints.write('{} {}\n'.format(typecode, game.kendallmod))
                
def main3():
    types  = [[i,j,k] for i in range(4) for j in range(4) for k in range(4) if
            i + j + k <= 3]
    games = {}
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    datapointsname = ''.join(["datapoints", timestr])
    with open(datapointsname, 'w') as datapoints:
        maxinfo = 0
        while maxinfo == 0:
            print("EXPERIMENT", i)
            print()
            entry = {}
            game = Game(payoffs())
            while game.payofftype != [0,0,0]: 
                game = Game(payoffs())
            game.maxinfo = game.info_in_equilibria() 
            datapoints.write('{} {}\n'.format(type_code(game.payofftype),
                game.maxinfo))
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["payoffs"] = game.payoffs
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            entry["type"] = game.payofftype
            games[i] = entry
            maxinfo = game.maxinfo
            print()
    gamesname = ''.join(["games", timestr])
    with open(gamesname, 'w') as gamefile:
        json.dump(games, gamefile)

def manygames():
    games = {}
    chances = [1/3, 1/3, 1/3]
    timestr = time.strftime("%d%b%H-%M")
    kendalls  = [2.00]
    for gametype in kendalls:
        print("Type: {}".format(gametype))
        for i in range(2000):
            print("EXPERIMENT", i)
            #print()
            entry = {}
            game = Game(payoffs())
            while round(game.kendalldistance, 2) != gametype: 
                game = Game(payoffs())
            game.equilibria, game.maxsinfo, game.maxrinfo = game.info_in_equilibria() 
            entry["equilibria"] = str(game.equilibria)
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["kendallmod"] = game.kendallmod
            entry["maxsinfo"] = game.maxsinfo
            entry["maxrinfo"] = game.maxrinfo
            games[str(game.payoffs)] = entry
        gamesname = ''.join(["type", '',str(gametype), timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(games, gamefile)


def manymanygames():
    games = {}
    chances = [1/3, 1/3, 1/3]
    for j in range(100):
        timestr = time.strftime("%d%b%H-%M")
        for i in range(200):
            print("EXPERIMENT", j,i)
            entry = {}
            game = Game(payoffs())
            game.equilibria, game.maxinfo = game.info_in_equilibria() 
            entry["equilibria"] = str(game.equilibria)
            entry["sender"] = game.sender
            entry["receiver"] = game.receiver
            entry["kendallmod"] = game.kendallmod
            entry["maxinfo"] = game.maxinfo
            games[str(game.payoffs)] = entry
        gamesname = ''.join(["manygames",str(j),str(i),"_",timestr])
        with open(gamesname, 'w') as gamefile:
            json.dump(games, gamefile)
