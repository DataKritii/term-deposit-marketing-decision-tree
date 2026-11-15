# Step-by-step: run the dashboard and get a live link

This guide takes you from the unzipped folder to a public link like
`https://term-deposit-targeting.streamlit.app` that anyone can open in a browser.

It has four parts: **A** run it on your laptop (about 10 minutes), **B** put it on GitHub (about 10 minutes),
**C** deploy it on Streamlit Community Cloud (about 10 minutes), **D** add the link to your README and resume.

Everything is free. You don't need to install any app other than Python and Git.

---

## Part A: Run it on your laptop

**A1. Check Python.** Open a terminal (Windows: *Anaconda Prompt* if you use Anaconda, otherwise *Command Prompt* or *PowerShell*; Mac: *Terminal*) and run:

```bash
python --version
```

You need Python 3.10 or newer. If you use Anaconda, you already have it.

**A2. Go to the project folder.** Use the path where you unzipped it, for example:

```bash
cd Documents/term-deposit-marketing-decision-tree
```

**A3. Create a virtual environment (recommended).** This keeps the project's packages separate from everything else on your computer.

```bash
python -m venv .venv
```

Then activate it:

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. Activate it again whenever you open a new terminal to work on this project.

> Using Anaconda instead? Run `conda create -n marketing python=3.12 -y` and then `conda activate marketing`.

**A4. Install the packages.**

```bash
pip install -r requirements-dev.txt
```

This installs everything for the dashboard, the notebook and the tests. It takes a few minutes the first time.

**A5. Run the dashboard.**

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501` with the dashboard. The first load takes a few seconds while the model trains.

- Move the sliders: the estimate updates instantly.
- Edit `app.py` and save: the page shows a **Rerun** button (or click **Always rerun** to update automatically).
- To stop the dashboard, click in the terminal and press `Ctrl + C`.

The first time, Streamlit may ask for your email in the terminal. Just press `Enter` to skip.

**A6. (Optional) Run the tests.**

```bash
pytest tests/
```

You should see `7 passed`. These check that the dashboard reproduces the notebook's results and that every screen works.

---

## Part B: Put the project on GitHub

**B1. Create a GitHub account** at [github.com](https://github.com) if you don't have one.

**B2. Install Git** from [git-scm.com/downloads](https://git-scm.com/downloads) (Mac users may already have it: check with `git --version`). Then, once only:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

**B3. Create an empty repository on GitHub.**
1. Click **+** (top right) → **New repository**.
2. Name: `term-deposit-marketing-decision-tree`
3. Visibility: **Public**. A public repo gives a public app, which is what you want for a portfolio. (Private repos also work, but the app is then private and recruiters can't open it without an invite.)
4. **Don't** tick "Add a README", ".gitignore" or "license". The project already has them.
5. Click **Create repository**.

**B4. Push the project.** In the terminal, inside the project folder:

```bash
git init
git add .
git commit -m "Term deposit targeting with decision trees, random forest and Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/term-deposit-marketing-decision-tree.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username.

The first push asks you to sign in. A browser window usually opens. If Git asks for a password instead, your normal GitHub password won't work. Create a *personal access token*: GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token**, tick **repo**, and paste the token as the password.

**B5. Check it.** Refresh the GitHub page. You should see the files, and the README displays below them with the charts.

---

## Part C: Deploy on Streamlit Community Cloud

**C1.** Go to [share.streamlit.io](https://share.streamlit.io) and click **Continue with GitHub**. Approve the permissions it asks for. Nothing is installed on your computer.

**C2.** Click **Create app** (top right), then choose **Deploy a public app from GitHub**.

**C3.** Fill in the form:

| Field | Value |
|---|---|
| Repository | `<your-username>/term-deposit-marketing-decision-tree` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | Choose a short name, e.g. `term-deposit-targeting` → `term-deposit-targeting.streamlit.app` |

**C4.** Open **Advanced settings** and set **Python version** to **3.12**. Click **Save**.

**C5.** Click **Deploy**. You'll see a log while it installs the packages from `requirements.txt` (2–5 minutes). When it finishes, the dashboard opens at your new link.

**C6. Test the live app.** Click through all three tabs and move a few sliders. The Model health tab takes a few extra seconds the first time, while the comparison and out-of-time test run.

### If something goes wrong

| Problem | Fix |
|---|---|
| Log says `ModuleNotFoundError` | A package is missing from `requirements.txt`. Add it, then commit and push; the app redeploys automatically |
| Repository not in the list | The repo must be public, and Streamlit needs permission to see it. Reconnect GitHub on share.streamlit.io |
| App shows an error page | Click **Manage app** (bottom right of the app) to see the log. The last red lines say what failed |

---

## Part D: Use the link

**D1. Add it to the README.** On GitHub, open `README.md`, click the pencil icon, and replace `YOUR-APP-URL` near the top with your link. Click **Commit changes**.

Or edit the file on your laptop and push:

```bash
git add README.md
git commit -m "Add live dashboard link"
git push
```

**D2. (Optional) Replace the screenshots.** To use your own screenshots in the README: open the live app, take a screenshot of each tab, save them over `reports/figures/dashboard_*.png` with the same file names, and push.

**D3. Add the link to your resume and LinkedIn**, for example:
*Term Deposit Campaign Targeting: decision tree and random forest models with campaign economics and an interactive Streamlit dashboard (live demo · GitHub)*

### Keeping it awake

On the free tier, an app goes to sleep after **12 hours** without visitors. The next visitor sees a **"Yes, get this app back up!"** button and waits about a minute while it restarts. **Open the link yourself shortly before an interview or before sending it to a recruiter.**

### Updating the app later

Every `git push` to `main` redeploys the app automatically. There's nothing to click.
