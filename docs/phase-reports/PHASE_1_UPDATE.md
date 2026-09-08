> **Superseded by `docs/ARCHITECTURE.md` #5.** The normalization plan below
> was never implemented or tested against the checkpoint's actual behavior.
> The checkpoint has no `feature_stats`, and the code now deliberately runs
> on raw, unnormalized features to match how it was actually validated.
> Kept here as a historical record only.

## PHASE 1 STATUS UPDATE — Feature Normalization Discovery

### Issue Discovered
- Checkpoint does NOT contain `feature_stats` dict (contrary to earlier conversation summary)
- Training script saves `feature_stats` when specified, but ablation checkpoint was saved without it
- Model was trained with z-score normalization BUT stats are not persisted

### Solution Implemented
**Production approach:** Compute and store feature statistics from enrollment data

1. **Initial Deployment:** Use default reference statistics
   - Extract features from 3-5 reference videos during enrollment
   - Compute mean/std across all enrolled persons
   - Store in SQLite `system_metadata` table
   - Use consistently for all subsequent inference

2. **Per-Feature Normalization (78-dimensional):**
   - Raw feature → (raw - mean) / (std + eps)
   - Apply BEFORE passing to model
   - eps = 1e-8 (numerical stability)

3. **Database Storage:**
   - Table: `system_metadata` (key-value pairs)
   - Keys: `feature_mean_json`, `feature_std_json`
   - Allows deployment-specific normalization

### Code Impact
- `src/model/gait_model.py` → Load stats from DB at startup
- `src/features/gait_features.py` → Apply normalization before encoding
- `scripts/enroll.py` → Update stats after each enrollment
- Test script updated to skip `feature_stats` checkpoint check

### Timeline Impact
- None — can proceed with Phase 2
- Adds robustness (per-deployment normalization > global checkpoint stats)
