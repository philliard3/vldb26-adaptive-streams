#[cfg(test)]
mod tests {
    use crate::scheduler::basic_probability_forecast::{History, PastData, PendingData};
    use crate::scheduler::{BinInfo, LabelRequirements, ShareableArray};
    use crossbeam::channel;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
    struct MockLabel;

    #[test]
    fn test_cleanup_stale_pending_excludes_purged() {
        // Setup
        let (tx, rx) = channel::unbounded();
        // BinInfo construction with correct fields
        let bins = vec![BinInfo {
            id: Some(MockLabel),
            valid_pipelines: ShareableArray::OwnedHeap(vec![1]),
            rewards: ShareableArray::OwnedHeap(vec![1.0]),
            costs: ShareableArray::OwnedHeap(vec![1.0]),
        }];

        let mut history = History::new(10, rx, bins);

        // --- Test 1: Default behavior (include purged items) ---
        history.exclude_purged_items_from_history = false;

        // Add a pending item
        let now = std::time::Instant::now();
        let pending = PendingData {
            tuple_id: 1,
            category: Some(MockLabel),
            time_of_creation_ns: 0,
            age_when_scheduling_ns: 0,
            time_of_scheduling: now - std::time::Duration::from_secs(2), // 2 seconds ago
        };
        history.pending_data.insert(1, pending);

        // Run cleanup with 1s limit (item is 2s old)
        // safety_limit_ns = 1_000_000_000
        history.cleanup_stale_pending(1_000_000_000);

        // Verify item was purged and ADDED to history
        assert!(
            history.pending_data.is_empty(),
            "Item 1 should be purged from pending"
        );
        assert_eq!(
            history.past_data.len(),
            1,
            "History should contain purged item 1"
        );
        assert_eq!(history.past_data[0].tuple_id, 1);
        assert_eq!(
            history.past_data[0].pipeline_id, 0,
            "Purged item should have pipeline_id 0"
        );

        // --- Test 2: Exclude behavior (skip purged items) ---
        history.past_data.clear();
        history.exclude_purged_items_from_history = true;

        // Add another pending item
        let pending2 = PendingData {
            tuple_id: 2,
            category: Some(MockLabel),
            time_of_creation_ns: 0,
            age_when_scheduling_ns: 0,
            time_of_scheduling: now - std::time::Duration::from_secs(2), // 2 seconds ago
        };
        history.pending_data.insert(2, pending2);

        // Run cleanup
        history.cleanup_stale_pending(1_000_000_000);

        // Verify item was purged but NOT added to history
        assert!(
            history.pending_data.is_empty(),
            "Item 2 should be purged from pending"
        );
        assert_eq!(
            history.past_data.len(),
            0,
            "History should NOT contain purged item 2"
        );
    }
}
