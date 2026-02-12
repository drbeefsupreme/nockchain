//! Git integration panel
//!
//! Provides UI for selecting git branches and commits for Docker image builds.

use std::path::{Path, PathBuf};

use egui::{RichText, Ui};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("Repository not found at {0}")]
    RepoNotFound(PathBuf),

    #[error("No branches found")]
    NoBranches,
}

/// Information about a git branch
#[derive(Debug, Clone)]
pub struct BranchInfo {
    /// Branch name
    pub name: String,

    /// Whether this is the current branch
    pub is_current: bool,

    /// Whether this is a remote branch
    pub is_remote: bool,

    /// Latest commit hash
    pub commit_hash: String,

    /// Latest commit message (first line)
    pub commit_message: String,
}

/// Information about a git commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Commit hash (short)
    pub hash: String,

    /// Full commit hash
    pub full_hash: String,

    /// Commit message (first line)
    pub message: String,

    /// Author name
    pub author: String,

    /// Commit time (as string)
    pub time: String,
}

/// Git repository information
pub struct GitRepo {
    repo: git2::Repository,
    path: PathBuf,
}

impl GitRepo {
    /// Open a git repository
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GitError> {
        let path = path.as_ref().to_path_buf();
        let repo = git2::Repository::open(&path).map_err(|e| {
            if e.code() == git2::ErrorCode::NotFound {
                GitError::RepoNotFound(path.clone())
            } else {
                GitError::Git(e)
            }
        })?;

        Ok(Self { repo, path })
    }

    /// Get the repository path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current branch name
    pub fn current_branch(&self) -> Result<String, GitError> {
        let head = self.repo.head()?;
        if head.is_branch() {
            Ok(head.shorthand().unwrap_or("HEAD").to_string())
        } else {
            // Detached HEAD - return commit hash
            let commit = head.peel_to_commit()?;
            Ok(format!("{}", &commit.id().to_string()[..7]))
        }
    }

    /// Get all local branches
    pub fn local_branches(&self) -> Result<Vec<BranchInfo>, GitError> {
        let mut branches = Vec::new();
        let current = self.current_branch()?;

        for branch_result in self.repo.branches(Some(git2::BranchType::Local))? {
            let (branch, _) = branch_result?;
            if let Some(name) = branch.name()? {
                let commit = branch.get().peel_to_commit()?;
                branches.push(BranchInfo {
                    name: name.to_string(),
                    is_current: name == current,
                    is_remote: false,
                    commit_hash: commit.id().to_string()[..7].to_string(),
                    commit_message: commit
                        .message()
                        .unwrap_or("")
                        .lines()
                        .next()
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }

        Ok(branches)
    }

    /// Get all remote branches
    pub fn remote_branches(&self) -> Result<Vec<BranchInfo>, GitError> {
        let mut branches = Vec::new();

        for branch_result in self.repo.branches(Some(git2::BranchType::Remote))? {
            let (branch, _) = branch_result?;
            if let Some(name) = branch.name()? {
                // Skip HEAD refs
                if name.ends_with("/HEAD") {
                    continue;
                }

                let commit = branch.get().peel_to_commit()?;
                branches.push(BranchInfo {
                    name: name.to_string(),
                    is_current: false,
                    is_remote: true,
                    commit_hash: commit.id().to_string()[..7].to_string(),
                    commit_message: commit
                        .message()
                        .unwrap_or("")
                        .lines()
                        .next()
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }

        Ok(branches)
    }

    /// Get recent commits on the current branch
    pub fn recent_commits(&self, count: usize) -> Result<Vec<CommitInfo>, GitError> {
        let mut commits = Vec::new();
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;

        for (i, oid_result) in revwalk.enumerate() {
            if i >= count {
                break;
            }

            let oid = oid_result?;
            let commit = self.repo.find_commit(oid)?;

            commits.push(CommitInfo {
                hash: oid.to_string()[..7].to_string(),
                full_hash: oid.to_string(),
                message: commit
                    .message()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string(),
                author: commit.author().name().unwrap_or("Unknown").to_string(),
                time: format_commit_time(commit.time()),
            });
        }

        Ok(commits)
    }

    /// Get commits for a specific branch
    pub fn commits_for_branch(
        &self,
        branch_name: &str,
        count: usize,
    ) -> Result<Vec<CommitInfo>, GitError> {
        let branch = self
            .repo
            .find_branch(branch_name, git2::BranchType::Local)
            .or_else(|_| self.repo.find_branch(branch_name, git2::BranchType::Remote))?;

        let commit = branch.get().peel_to_commit()?;

        let mut commits = Vec::new();
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push(commit.id())?;

        for (i, oid_result) in revwalk.enumerate() {
            if i >= count {
                break;
            }

            let oid = oid_result?;
            let c = self.repo.find_commit(oid)?;

            commits.push(CommitInfo {
                hash: oid.to_string()[..7].to_string(),
                full_hash: oid.to_string(),
                message: c
                    .message()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string(),
                author: c.author().name().unwrap_or("Unknown").to_string(),
                time: format_commit_time(c.time()),
            });
        }

        Ok(commits)
    }
}

/// Format a commit time for display
fn format_commit_time(time: git2::Time) -> String {
    use chrono::{TimeZone, Utc};
    let dt = Utc.timestamp_opt(time.seconds(), 0);
    match dt {
        chrono::LocalResult::Single(dt) => dt.format("%Y-%m-%d %H:%M").to_string(),
        _ => "Unknown".to_string(),
    }
}

/// Git selection panel
pub struct GitPanel {
    /// Repository (if open)
    repo: Option<GitRepo>,

    /// Error message (if any)
    error: Option<String>,

    /// Path to repository
    repo_path: String,

    /// Local branches
    local_branches: Vec<BranchInfo>,

    /// Remote branches
    remote_branches: Vec<BranchInfo>,

    /// Recent commits for selected branch
    commits: Vec<CommitInfo>,

    /// Selected branch
    selected_branch: Option<String>,

    /// Selected commit
    selected_commit: Option<String>,

    /// Whether to show remote branches
    show_remote: bool,

    /// Number of commits to show
    commit_count: usize,
}

impl Default for GitPanel {
    fn default() -> Self {
        Self {
            repo: None,
            error: None,
            repo_path: ".".to_string(),
            local_branches: Vec::new(),
            remote_branches: Vec::new(),
            commits: Vec::new(),
            selected_branch: None,
            selected_commit: None,
            show_remote: false,
            commit_count: 20,
        }
    }
}

impl GitPanel {
    /// Create a new git panel
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a repository
    pub fn open_repo(&mut self, path: impl AsRef<Path>) {
        self.repo_path = path.as_ref().to_string_lossy().to_string();

        match GitRepo::open(&self.repo_path) {
            Ok(repo) => {
                // Load branches
                self.local_branches = repo.local_branches().unwrap_or_default();
                self.remote_branches = repo.remote_branches().unwrap_or_default();

                // Select current branch
                self.selected_branch = repo.current_branch().ok();

                // Load commits
                self.commits = repo.recent_commits(self.commit_count).unwrap_or_default();

                self.repo = Some(repo);
                self.error = None;
            }
            Err(e) => {
                self.error = Some(e.to_string());
                self.repo = None;
            }
        }
    }

    /// Get the selected git reference (branch or commit)
    pub fn selected_ref(&self) -> Option<String> {
        self.selected_commit
            .clone()
            .or_else(|| self.selected_branch.clone())
    }

    /// Refresh the repository
    pub fn refresh(&mut self) {
        if let Some(ref repo) = self.repo {
            self.local_branches = repo.local_branches().unwrap_or_default();
            self.remote_branches = repo.remote_branches().unwrap_or_default();

            if let Some(ref branch) = self.selected_branch {
                self.commits = repo
                    .commits_for_branch(branch, self.commit_count)
                    .unwrap_or_else(|_| repo.recent_commits(self.commit_count).unwrap_or_default());
            } else {
                self.commits = repo.recent_commits(self.commit_count).unwrap_or_default();
            }
        }
    }

    /// Render the panel
    pub fn show(&mut self, ui: &mut Ui) -> GitPanelResponse {
        let mut response = GitPanelResponse::default();

        // Repository path
        ui.horizontal(|ui| {
            ui.label("Repository:");
            if ui.text_edit_singleline(&mut self.repo_path).changed() {
                // Will reopen on "Open" click
            }
            if ui.button("Open").clicked() {
                let path = self.repo_path.clone();
                self.open_repo(&path);
                response.repo_opened = self.repo.is_some();
            }
            if self.repo.is_some() && ui.button("Refresh").clicked() {
                self.refresh();
                response.refreshed = true;
            }
        });

        // Error display
        if let Some(ref error) = self.error {
            ui.colored_label(egui::Color32::RED, error);
            return response;
        }

        if self.repo.is_none() {
            ui.label("No repository open");
            return response;
        }

        ui.separator();

        // Branch selection
        ui.horizontal(|ui| {
            ui.label(RichText::new("Branch:").strong());
            ui.checkbox(&mut self.show_remote, "Show remote");
        });

        // Branch list
        let branches = if self.show_remote {
            self.local_branches
                .iter()
                .chain(self.remote_branches.iter())
                .collect::<Vec<_>>()
        } else {
            self.local_branches.iter().collect::<Vec<_>>()
        };

        egui::ScrollArea::vertical()
            .id_salt("git_branches_scroll")
            .max_height(150.0)
            .show(ui, |ui| {
                for branch in branches {
                    let is_selected = self.selected_branch.as_ref() == Some(&branch.name);
                    let label = if branch.is_current {
                        format!("* {} ({})", branch.name, branch.commit_hash)
                    } else if branch.is_remote {
                        format!("  {} ({})", branch.name, branch.commit_hash)
                    } else {
                        format!("  {} ({})", branch.name, branch.commit_hash)
                    };

                    if ui.selectable_label(is_selected, label).clicked() {
                        self.selected_branch = Some(branch.name.clone());
                        self.selected_commit = None;

                        // Load commits for this branch
                        if let Some(ref repo) = self.repo {
                            self.commits = repo
                                .commits_for_branch(&branch.name, self.commit_count)
                                .unwrap_or_default();
                        }

                        response.branch_selected = Some(branch.name.clone());
                    }
                }
            });

        ui.separator();

        // Commit selection
        ui.label(RichText::new("Commit:").strong());

        egui::ScrollArea::vertical()
            .id_salt("git_commits_scroll")
            .max_height(200.0)
            .show(ui, |ui| {
                for commit in &self.commits {
                    let is_selected = self.selected_commit.as_ref() == Some(&commit.hash);
                    let label = format!(
                        "{} {} - {}",
                        commit.hash,
                        commit.time,
                        truncate(&commit.message, 50)
                    );

                    if ui.selectable_label(is_selected, label).clicked() {
                        self.selected_commit = Some(commit.hash.clone());
                        response.commit_selected = Some(commit.hash.clone());
                    }
                }
            });

        // Selected summary
        ui.separator();
        if let Some(ref commit) = self.selected_commit {
            ui.label(format!("Selected: commit {}", commit));
        } else if let Some(ref branch) = self.selected_branch {
            ui.label(format!("Selected: branch {}", branch));
        } else {
            ui.label("No selection");
        }

        response
    }
}

/// Truncate a string
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

/// Response from the git panel
#[derive(Default)]
pub struct GitPanelResponse {
    /// Whether a repository was opened
    pub repo_opened: bool,

    /// Whether the repository was refreshed
    pub refreshed: bool,

    /// Branch that was selected
    pub branch_selected: Option<String>,

    /// Commit that was selected
    pub commit_selected: Option<String>,
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn init_test_repo(path: &Path) -> git2::Repository {
        let repo = git2::Repository::init(path).unwrap();

        // Create a dummy commit
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        let tree_id = {
            let mut index = repo.index().unwrap();
            index.write_tree().unwrap()
        };

        {
            let tree = repo.find_tree(tree_id).unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
                .unwrap();
        }

        repo
    }

    #[test]
    fn test_git_repo_open() {
        let dir = tempdir().unwrap();
        init_test_repo(dir.path());

        let repo = GitRepo::open(dir.path()).unwrap();
        assert_eq!(repo.path(), dir.path());
    }

    #[test]
    fn test_git_repo_current_branch() {
        let dir = tempdir().unwrap();
        init_test_repo(dir.path());

        let repo = GitRepo::open(dir.path()).unwrap();
        // Default branch is usually "master" or "main"
        let branch = repo.current_branch().unwrap();
        assert!(!branch.is_empty());
    }

    #[test]
    fn test_git_repo_local_branches() {
        let dir = tempdir().unwrap();
        init_test_repo(dir.path());

        let repo = GitRepo::open(dir.path()).unwrap();
        let branches = repo.local_branches().unwrap();

        assert!(!branches.is_empty());
        assert!(branches.iter().any(|b| b.is_current));
    }

    #[test]
    fn test_git_repo_recent_commits() {
        let dir = tempdir().unwrap();
        init_test_repo(dir.path());

        let repo = GitRepo::open(dir.path()).unwrap();
        let commits = repo.recent_commits(10).unwrap();

        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].message, "Initial commit");
    }

    #[test]
    fn test_git_panel_default() {
        let panel = GitPanel::new();
        assert!(panel.repo.is_none());
        assert!(panel.selected_ref().is_none());
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long string", 10), "this is...");
    }

    #[test]
    fn test_format_commit_time() {
        let time = git2::Time::new(1700000000, 0);
        let formatted = format_commit_time(time);
        assert!(formatted.contains("2023"));
    }
}
