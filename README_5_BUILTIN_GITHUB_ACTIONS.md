# BurTorch: Builtin GitHub Actions

----

The goal of this document is to describe the extra possibilities of using this project to leverage GitHub Continuous Integration (CI).

## Background

The GitHub itself has the means for organized Continuous Integration (CI). The CI is an engineering practice where developers frequently launch automatic checks, tests, and builds. We provide the means to activate GitHub CI by providing custom-written configuration files.
The GitHub actions are activated via changes (commits) pushed via Git into the GitHub repository. GitHub uses its own terminology, and it is worthwhile to summarize it next.

## GitHub Terminology for Continuous Integration


* *Workflow* - is a configurable automated process that will run one or more jobs.

* *Event*  - is a specific activity in a repository that triggers a workflow run.

* *Runner* - is a server that runs your workflows when they're triggered. Each runner can run a single job at a time. GitHub provides Ubuntu Linux, Microsoft Windows, and macOS runners.

* *Job* - is a set of steps in a workflow that executes on the (same) runner.

* *Step* - is either a bash/shell script that will be executed or an action that will be run.

* *Action* - is a custom application for the GitHub Actions platform that performs a complex but frequently repeated task.

# GitGub Documentation on the Subject

[1] https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions

[2] https://docs.github.com/en/actions/using-workflows/about-workflows

[3] https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources

[4] Environment variables: https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables

[5] Github runners are available on GitHub: https://github.com/actions/runner-images#available-images

## Activate CI action after your Commit

To activate CI, please add the following text to the commit message:

| If the commit message contains | Action                                                                                              |
|----------------------------|-----------------------------------------------------------------------------------------------------|
| pls.build                  | Make build and launch unit tests and regression tests on Linux, Windows, macOS x64, and macOS aarch64.  |
| pls.pack                   | Prepare artifacts from the build in GitHub to download them.                                       | 
| pls.info                   | Print information about the virtual or physical host used to build the project in CI.                 |
| pls.clean-github           | Clean up Github from old workflows. Builds (and their artifacts) older than 15 days will be removed. | 
| pls.check-src              | Run various tools that check the source code. Runs cloc and cppcheck.                              | 

The configuration files itself is located in the project folder:

```bash
./.github/workflows/ci.yaml
```

If the project is cloned to activate runners, you need to perform the needed configuration in the:
```
https://github.com/<github-user>/<repo-name>/settings/actions
```

# Why we have used CI

* **Automated Testing:** Ensures the code works as expected with each change.
* **Consistent Builds:** By automating the build, we did not need to manually compile or build the code across three Operating Systems.
