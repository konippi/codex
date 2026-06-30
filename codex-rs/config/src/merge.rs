use crate::key_aliases::normalize_key_aliases;
use crate::key_aliases::normalized_with_key_aliases;
use codex_network_proxy::normalize_host;
use toml::Value as TomlValue;

/// Merge config `overlay` into `base`, giving `overlay` precedence.
pub fn merge_toml_values(base: &mut TomlValue, overlay: &TomlValue) {
    merge_toml_values_at_path(base, overlay, &mut Vec::new());
}

fn merge_toml_values_at_path(base: &mut TomlValue, overlay: &TomlValue, path: &mut Vec<String>) {
    if let TomlValue::Table(overlay_table) = overlay
        && let TomlValue::Table(base_table) = base
    {
        normalize_key_aliases(path, base_table);
        let mut overlay_table = overlay_table.clone();
        normalize_key_aliases(path, &mut overlay_table);
        if is_permission_network_domains_path(path) {
            normalize_network_domain_keys(base_table);
            normalize_network_domain_keys(&mut overlay_table);
        }

        for (key, value) in overlay_table {
            path.push(key.clone());
            if let Some(existing) = base_table.get_mut(&key) {
                merge_toml_values_at_path(existing, &value, path);
            } else {
                base_table.insert(key, normalized_with_key_aliases(&value, path));
            }
            path.pop();
        }
    } else {
        *base = normalized_with_key_aliases(overlay, path);
    }
}

fn is_permission_network_domains_path(path: &[String]) -> bool {
    matches!(
        path,
        [permissions, _, network, domains]
            if permissions == "permissions" && network == "network" && domains == "domains"
    )
}

fn normalize_network_domain_keys(table: &mut toml::map::Map<String, TomlValue>) {
    let entries = std::mem::take(table);
    for (pattern, value) in entries {
        table.insert(normalize_host(&pattern), value);
    }
}

/// Remove the value at nested `path`, pruning tables that become empty, and
/// return the removed value (if any).
pub(crate) fn remove_nested_field_and_prune_empty(
    value: &mut TomlValue,
    path: &[&str],
) -> Option<TomlValue> {
    let (key, remaining) = path.split_first()?;
    let table = value.as_table_mut()?;
    if remaining.is_empty() {
        return table.remove(*key);
    }

    let removed = table
        .get_mut(*key)
        .and_then(|child| remove_nested_field_and_prune_empty(child, remaining));
    if table
        .get(*key)
        .and_then(TomlValue::as_table)
        .is_some_and(toml::map::Map::is_empty)
    {
        table.remove(*key);
    }
    removed
}

#[cfg(test)]
#[path = "merge_tests.rs"]
mod tests;
