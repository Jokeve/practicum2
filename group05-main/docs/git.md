# Git for this lab

Git is optional. You can submit the entire updated repository as a ZIP on Ufora and receive the full base grade. To earn the **one-point Git bonus**, make at least **one commit and one push** to your private GitLab project and submit its link. We encourage you to save intermediate work with Git after each experiment.

## The three ideas you need

- A **commit** saves a named snapshot of your changes **on your computer**. It does not upload anything.
- A **push** uploads your local commits to your **remote** project on GitLab.
- A **pull** brings newer commits from that remote into your local copy. Use it before editing on a second computer or when your project has changed online.

A **repository** is the folder Git tracks. A **branch** is a line of commits; you can do this lab on the existing `main` branch. `origin` is Git's usual name for your remote project. `git status` tells you your branch, changed files, and whether anything is waiting to be committed.

## Start from your private project

If you choose the Git route, accept your GitLab invitation and create an account if prompted. Do the following on the **cluster login node**, where you will clone and push. Check whether you already have a public SSH key:

```bash
ls ~/.ssh/id_ed25519.pub
```

If that file does not exist, create a key pair with your email address as its label:

```bash
ssh-keygen -t ed25519 -C "your.name@ugent.be"
```

Accept the default location and choose a passphrase. Show the **public** key with `cat ~/.ssh/id_ed25519.pub`, then copy that entire line into GitLab: your avatar → **Edit profile** → **Access → SSH keys** → **Add new key**. Never share `~/.ssh/id_ed25519` (the private key) or place it in this repository. If you already have a working key under another filename, use its `.pub` file instead. GitLab's [SSH key guide](https://docs.gitlab.com/user/ssh/) has further details.

Test the connection on the cluster login node:

```bash
ssh -T git@gitlab.com
```

You should see a GitLab welcome message. On the first connection, check GitLab's host fingerprint before accepting the host prompt, as described in the [SSH key guide](https://docs.gitlab.com/user/ssh/). One key on your cluster account works for the private GitLab projects that your GitLab account can access; you do not need a key per project.

Open **your own** private project in GitLab and copy its **SSH** Clone URL. Replace the example URL below with yours and change `group01` to your group number:

```bash
cd /data/gent/courses/2026/ASIP_E092841/groups/group01
git clone git@gitlab.com:COURSE/submissions/YOUR_PROJECT.git nn-unet-lab
cd nn-unet-lab
git status
```

`clone` downloads the starter repository and sets `origin` to your project. Later, `git pull` updates that clone and `git push` uploads your commits using the same SSH key. Keep the BraTS data outside this folder. If Git asks for your name and email before the first commit, set them with `git config --global user.name "Your Name"` and `git config --global user.email "your.name@ugent.be"`.

## Save and upload your work

After an experiment, check what changed, add the files you mean to save, make a commit, and push it:

```bash
git status
git add src/data_loader.py README.md
git diff --cached --stat
git commit -m "Add training augmentations"
git push
```

Change the file names and message to match your work. `git add` places files in the **staging area** for the next commit. `git diff --cached --stat` gives a short review of what will be committed. `git commit` saves that staged snapshot locally; `git push` uploads it. A single commit and push of your final work is enough for the bonus. Earlier commits make it easier to return to a working version and show how your ablation progressed.

If you work on another computer, begin with `git pull` in your local repository, then edit and commit. Also pull before pushing if Git tells you the remote has newer commits. If Git reports a merge conflict, open the named file, choose the correct content, then add and commit the resolved file. Ask a teaching assistant if you are unsure which version to keep.

## Good habits

- Commit one understandable change at a time, such as one lab step. Use a short message that says what changed: `Add deep supervision loss` is clearer than `updates`.
- Run `git status` before and after a commit. Review staged files with `git diff --cached --stat`, or `git diff --cached` to see the exact lines.
- Push after useful commits so your remote copy is backed up. Pull before working on a different computer.
- Keep images, masks, checkpoints, W&B logs, and secrets out of Git. `.gitignore` excludes `data/`, `runs/`, `wandb/`, and the personal `.wandb-project` file. The tracked `jobs/submission.sh` contains no API key; run `bash setup-wandb.sh` on the cluster instead.
- Include `reports/ablation-report.pdf` in the final repository. Check that the PDF is staged before your final commit.

## Submit

For the **Git route**, make sure `git status` is clean after your final commit and push. Submit your private GitLab project URL and live W&B Report URL. The project must include the whole updated repository and `reports/ablation-report.pdf`.

For the **Ufora route**, ZIP the whole updated repository, including `jobs/submission.sh` and the report PDF, but leave out data, `wandb/`, `runs/`, `.wandb-project`, and `.git/`. Upload the ZIP and provide your live W&B Report URL. This route does not earn the Git bonus.
