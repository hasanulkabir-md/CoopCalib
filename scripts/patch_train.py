"""
patch_train.py — Run from C:\CoopCalib\TUTR\
Patches train.py in-place with all four aggressive variant changes.
Usage: python ..\scripts\patch_train.py
"""
import os
import shutil

train_path = os.path.join(os.getcwd(), 'train.py')
backup_path = os.path.join(os.getcwd(), 'train_ORIGINAL_BACKUP.py')

# Safety backup
if not os.path.exists(backup_path):
    shutil.copy2(train_path, backup_path)
    print(f'Backup saved: {backup_path}')
else:
    print(f'Backup already exists: {backup_path}')

with open(train_path, 'r', encoding='utf-8') as f:
    src = f.read()

original_src = src
changes_made = []

# ------------------------------------------------------------------
# CHANGE 1 — add --lambda3 and --warmup_epoch arguments
# ------------------------------------------------------------------
old1 = ("parser.add_argument('--lambda2', type=float, default=0.0,\n"
        "                    help='Weight for L_energy social margin loss (V2)')")
new1 = (old1 +
        "\nparser.add_argument('--lambda3', type=float, default=0.0,\n"
        "                    help='Weight for L_RankingSVR ranking-aware social loss (V2R)')\n"
        "parser.add_argument('--warmup_epoch', type=int, default=0,\n"
        "                    help='Epoch after which L_ECE activates (V1W). 0 = no warmup.')")

if old1 in src:
    src = src.replace(old1, new1, 1)
    changes_made.append('CHANGE 1: lambda3 + warmup_epoch arguments added')
else:
    print('WARNING: CHANGE 1 target not found — check lambda2 argument text in train.py')

# ------------------------------------------------------------------
# CHANGE 2a — add LossRankingSVR to import line
# ------------------------------------------------------------------
old2a = 'from metrics.loss_functions import LossECE, LossEnergy'
new2a = 'from metrics.loss_functions import LossECE, LossEnergy, LossRankingSVR'

if old2a in src:
    src = src.replace(old2a, new2a, 1)
    changes_made.append('CHANGE 2a: LossRankingSVR added to import')
else:
    print('WARNING: CHANGE 2a target not found — check import line in train.py')

# ------------------------------------------------------------------
# CHANGE 2b — instantiate ranking_svr_criterion
# ------------------------------------------------------------------
old2b = 'energy_criterion = LossEnergy(r_ped=0.3).cuda()'
new2b = ('energy_criterion      = LossEnergy(r_ped=0.3).cuda()\n'
         'ranking_svr_criterion = LossRankingSVR(r_ped=0.3, temperature=5.0).cuda()')

if old2b in src:
    src = src.replace(old2b, new2b, 1)
    changes_made.append('CHANGE 2b: ranking_svr_criterion instantiated')
else:
    print('WARNING: CHANGE 2b target not found — check energy_criterion line in train.py')

# ------------------------------------------------------------------
# CHANGE 3 — add warmup gate to lambda1 block
# ------------------------------------------------------------------
old3 = 'if args.lambda1 > 0.0:'
new3 = 'if args.lambda1 > 0.0 and epoch >= args.warmup_epoch:'

if old3 in src:
    src = src.replace(old3, new3, 1)
    changes_made.append('CHANGE 3: warmup gate added to lambda1 block')
else:
    print('WARNING: CHANGE 3 target not found — check lambda1 if-block in train.py')

# ------------------------------------------------------------------
# CHANGE 4 — add lambda3 block before loss.backward()
# Find the LAST occurrence of loss.backward() inside train()
# which is the one inside the training loop
# ------------------------------------------------------------------
lambda3_block = (
    '        # CoopCalib V2R - LossRankingSVR ranking-aware social loss\n'
    '        # Penalises top-ranked candidate for social violations\n'
    '        # Gradient flows into CLF-FC head - directly targets SVR\n'
    '        if args.lambda3 > 0.0:\n'
    '            if args.lambda1 <= 0.0 and args.lambda2 <= 0.0:\n'
    '                pred_traj_k, scores_k = model(ped_obs, neis_obs, motion_modes,\n'
    '                                              mask, None, test=True)\n'
    '            pred_traj_2d_r = pred_traj_k.reshape(\n'
    '                pred_traj_k.shape[0], pred_traj_k.shape[1], -1, 2)\n'
    '            scores_2d_r   = scores_k.squeeze(-1) if scores_k.dim() == 3 else scores_k\n'
    '            nei_futures_r = neis[:, :, args.obs_len:, :]\n'
    '            loss = loss + args.lambda3 * ranking_svr_criterion(\n'
    '                pred_traj_2d_r, scores_2d_r, nei_futures_r)\n'
    '\n'
)

# Target: the loss.backward() line inside train()
# We insert the lambda3 block immediately before it
old4 = '        loss.backward()\n        optimizer.step()\n        optimizer.zero_grad()\n        total_loss.append(loss.item())'
new4 = lambda3_block + old4

if old4 in src:
    src = src.replace(old4, new4, 1)
    changes_made.append('CHANGE 4: lambda3 block added before loss.backward()')
else:
    print('WARNING: CHANGE 4 target not found — check loss.backward() block in train.py')

# ------------------------------------------------------------------
# Write patched file
# ------------------------------------------------------------------
if src == original_src:
    print('\nERROR: No changes were made. Check warnings above.')
    print('Original train.py is unchanged.')
else:
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(src)
    print(f'\nPatched train.py written successfully.')
    print(f'Changes applied ({len(changes_made)}/4):')
    for c in changes_made:
        print(f'  [OK] {c}')

# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------
print('\nVerification — searching for patched terms:')
terms = ['lambda3', 'warmup_epoch', 'LossRankingSVR', 'ranking_svr_criterion']
all_found = True
for term in terms:
    count = src.count(term)
    status = 'OK' if count > 0 else 'MISSING'
    print(f'  [{status}] {term}: {count} occurrence(s)')
    if count == 0:
        all_found = False

if all_found and len(changes_made) == 4:
    print('\nAll 4 changes applied successfully.')
    print('Next: python train.py --help | findstr /i "lambda3 warmup"')
    print('Then: python ..\\scripts\\run_aggressive.py')
else:
    print('\nSome changes missing — check warnings above.')
    print('Restore backup if needed:')
    print('  copy train_ORIGINAL_BACKUP.py train.py')
