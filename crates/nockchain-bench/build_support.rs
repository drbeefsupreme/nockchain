pub fn tracked_git_watch_paths(
    head_path: &str,
    packed_refs_path: &str,
    head_ref_path: Option<&str>,
) -> Vec<String> {
    let mut paths = vec![head_path.to_string(), packed_refs_path.to_string()];
    if let Some(head_ref_path) = head_ref_path {
        let head_ref_path = head_ref_path.trim();
        if !head_ref_path.is_empty() {
            paths.push(head_ref_path.to_string());
        }
    }
    paths
}
