# Getting Started

Thank you so much for your consideration of contributing to AutoRAG open source project, and a big welcome! Users in our community are the ones who make it a reality—people just like you.

By reading and adhering to these principles, we can ensure that the contribution process is simple and efficient for all parties. Additionally, it conveys your agreement to honour the developers' time as they oversee and work on these open-source projects. We will respect you in return by taking care of your problem, evaluating your changes, and assisting you in completing your pull requests.


- View the [README](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/README.md) or [watch this video](https://youtu.be/2ojK8xjyXAU?si=nJz-IgrXFaMiyyW5) to get your development environment up and running. 
- Learn how to [format pull requests](#submitting-a-pull-request).
- Read how to [rebase/merge upstream branches](#configuring-remotes).
- Follow our [code of conduct](CODE_OF_CONDUCT.md).
- [Find an issue to work on](https://github.com/Marker-Inc-Korea/AutoRAG/issues) and start smashing!
  
# Contributing Guidelines [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Marker-Inc-Korea/AutoRAG/issues))

When contributing to this repository, please first discuss the change you wish to make via an issue.

Remember that this is an inclusive community, committed to creating a safe, positive environment. See the whole [Code of Conduct](CODE_OF_CONDUCT.md) and please follow it in all your interactions with the project.


## Submitting or Requesting an Issue/Enhancement

### The best ways to report issues or make requests for improvements are as follows:
- Please explore the issue tracker before submitting an issue. There may already be an issue for your issue, and the conversation may have made remedies readily apparent.
- When creating the problem, include the screenshots also.

### Best Practices for getting assigned to work on an Issue/Enhancement:
- If you would like to work on an issue, inform in the issue ticket by commenting on it.
- Please be sure that you are able to reproduce the issue, before working on it. If not, please ask for clarification by commenting or asking the issue creator.

**Note:** Please do not work on an issue which is already being worked on by another contributor. We don't encourage creating multiple pull requests for the same issue. Also, please allow the assigned person at least 2 days to work on the issue (The time might vary depending on the difficulty). If there is no progress after the deadline, please comment on the issue asking the contributor whether he/she is still working on it. If there is no reply, then feel free to work on the issue.


## Submitting a Pull Request

### Best Practices to send Pull Requests:
  - Fork the [project](https://github.com/Marker-Inc-Korea/AutoRAG) on GitHub
  - Clone the project locally into your system.
```
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
```
  - Make sure you are in the `main` branch.
```
git checkout main
```
  - Create a new branch with a meaningful name before adding and committing your changes.
```
git checkout -b branch-name
```
  - Add the files you changed. (avoid using `git add .`)
```
git add file-name
```
  - Commit the added files
```
git commit
```
  - If you forgot to add some changes, you can edit your previous commit message.
```
git commit --amend
```
  - Squash multiple commits to a single commit. (example: squash last two commits done on this branch into one)
```
git rebase --interactive HEAD~2 
```
  - Push this branch to your remote repository on GitHub.
```
git push origin branch-name
```
  - If any of the squashed commits have already been pushed to your remote repository, you need to do a force push.
```
git push origin remote-branch-name --force
```
  - Follow the Pull request template and submit a pull request with a motive for your change and the method you used to achieve it to be merged with the `main` branch.
  - If you can, please submit the pull request with the fix or improvements including tests.
  - During review, if you are requested to make changes, rebase your branch and squash the multiple commits into one. Once you push these changes the pull request will edit automatically.


## Configuring remotes
When a repository is cloned, it has a default remote called `origin` that points to your fork on GitHub, not the original repository it was forked from. To keep track of the original repository, you should add another remote called `upstream`.

1. Set the `upstream`.
```
git remote add upstream https://github.com/Marker-Inc-Korea/AutoRAG.git
```
2. Use `git remote -v` to check the status. The output must be something like this:
```
  > origin    https://github.com/your-username/AutoRAG.git (fetch)
  > origin    https://github.com/your-username/AutoRAG.git (push)
  > upstream  https://github.com/Marker-Inc-Korea/AutoRAG.git (fetch)
  > upstream  https://github.com/Marker-Inc-Korea/AutoRAG.git (push)
```
3. To update your local copy with remote changes, run the following: (This will give you an exact copy of the current remote. You should not have any local changes on your main branch, if you do, use rebase instead).
```
git fetch upstream
git checkout main
git merge upstream/main
```
4. Push these merged changes to the main branch on your fork. Ensure to pull in upstream changes regularly to keep your forked repository up to date.
```
git push origin main
```
5. Switch to the branch you are using for some piece of work.
```
git checkout branch-name
```
6. Rebase your branch, which means, take in all latest changes and replay your work in the branch on top of this - this produces cleaner versions/history.
```
git rebase main
```
7. Push the final changes when you're ready.
```
git push origin branch-name
```

## After your Pull Request is merged
After your pull request is merged, you can safely delete your branch and pull the changes from the main (upstream) repository.

1. Delete the remote branch on GitHub.
```
git push origin --delete branch-name
```
2. Checkout the main branch.
```
git checkout main
```
3. Delete the local branch.
```
git branch -D branch-name
```
4. Update your main branch with the latest upstream version.
```
git pull upstream main
```
