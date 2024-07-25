## How to contribute

1. Fork the project.
2. Clone your fork to your computer.
    * From the command line: `git clone https://github.com/<USERNAME>/evoke.git`
3. Change into your new project folder.
    * From the command line: `cd evoke`
4. [optional]  Add the upstream repository to your list of remotes.
    * From the command line: `git remote add upstream https://github.com/signalling-games-org/evoke.git`
5. Create a branch for your new feature.
    * From the command line: `git checkout -b my-feature-branch-name`
6. Make your changes.
    * Avoid making changes to more files than necessary for your feature (i.e. refrain from combining your "real" merge request with incidental bug fixes). This will simplify the merging process and make your changes clearer.
    * If you are adding examples from the literature:
        * place the file containing the example into `examples`, naming it `<first author><year><title's first letter>.py` in lower case without spaces, e.g. `skyrms2010signals.py`.
        * each example object should inherit from an existing Figure object. If necessary create your own in `figure.py` first.
        * name each example object <First author><year>_<figure number> in camel case with underscores replacing periods, e.g. Skyrms's Figure 3.4 becomes `Skyrms2010_3_4`. If the figure does not have a number in the original text, use a uniquely identifying and descriptive name.
7. Commit your changes. From the command line:
    * `git add <FILE-NAMES>`
    * `git commit -m "A descriptive commit message"`
8. While you were working some other changes might have gone in and break your stuff or vice versa. This can be a *merge conflict* but also conflicting behavior or code. Before you test, merge with master.
    * `git fetch upstream`
    * `git merge upstream/main`
9. Test. Run the program and do something related to your feature/fix.
10. Push the branch, uploading it to GitHub.
    * `git push origin my-feature-branch-name`
11. Make a "merge request" from your branch here on GitHub.

## How to run tests

Test scripts can be found in `evoke/tests/`.
Each test script corresponds to a source script; for example `test_evolve.py` contains tests for the classes and methods in `evolve.py`.
To run a test script simply call `python <script name>`; for example `python test_evolve.py` will run all the tests for `evolve.py`.
To run all test scripts at once, navigate to the test directory and run `python -m unittest`.

## How to report issues or problems

Please use the [issue tracker](https://github.com/signalling-games-org/evoke/issues) to report problems.

## How to seek support

Email [Stephen Mann](mailto:stephenfmann@gmail.com).

## Attribution

Parts of this document were adapted from [taguette](https://gitlab.com/remram44/taguette/-/tree/master)'s [CONTRIBUTING.md](https://gitlab.com/remram44/taguette/-/blob/master/CONTRIBUTING.md).
