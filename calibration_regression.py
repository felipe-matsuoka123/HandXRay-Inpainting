import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

gt_path        = r'C:\Projetos\hand-bone\Bone age ground truth.xlsx'
pred_orig_path = r'C:\Projetos\hand-bone\boneage_predictions.csv'
pred_clean_path= r'C:\Projetos\hand-bone\cleaned_boneage_predictions.csv'
save_path      = r'C:\Projetos\hand-bone\figs\calibration_fit_vs_gt.png'

gt_df        = pd.read_excel(gt_path)
pred_orig_df = pd.read_csv(pred_orig_path)
pred_clean_df= pd.read_csv(pred_clean_path)

merged = (gt_df
          .merge(pred_orig_df[['Case ID','Predicted Months']].rename(columns={'Predicted Months':'Pred_orig'}), on='Case ID')
          .merge(pred_clean_df[['Case ID','Predicted Months']].rename(columns={'Predicted Months':'Pred_clean'}), on='Case ID'))


y_true  = merged['Ground truth bone age (months)']
y_clean = merged['Pred_clean']

slope, intercept, *_ = stats.linregress(y_clean, y_true)
y_cal = intercept + slope * y_clean

def metrics(err):
    mae  = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    bias = err.mean()
    return mae, rmse, bias

err_raw = y_clean - y_true
err_cal = y_cal   - y_true

mae_raw, rmse_raw, bias_raw = metrics(err_raw)
mae_cal, rmse_cal, bias_cal = metrics(err_cal)

summary = pd.DataFrame({
    'Metric': ['MAE (months)', 'RMSE (months)', 'Bias (months)'],
    'Cleaned – Raw'      : [mae_raw, rmse_raw, bias_raw],
    'Cleaned – Calibrated': [mae_cal, rmse_cal, bias_cal]
})
print(summary)

sns.set_theme(style="whitegrid", font_scale=1.2)
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(y_true, y_cal, alpha=0.6, edgecolor='k', linewidth=0.3, s=60)

lims = [min(y_true.min(), y_cal.min()), max(y_true.max(), y_cal.max())]
ax.plot(lims, lims, ls='--', color='grey', linewidth=1.5)
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('Ground truth (months)')
ax.set_ylabel('Calibrated prediction (months)')
ax.set_title('Calibration Fit: GT vs. Calibrated Prediction')

ax.text(0.05, 0.95, f'MAE = {mae_cal:.2f} m',
        transform=ax.transAxes,
        fontsize=12, bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment="top")

caption_text = (
    f"Figure 3. Calibrated bone age predictions vs. ground truth. "
    f"The dashed line indicates perfect agreement. Calibration removes the mean bias, "
    f"but prediction errors remain substantial (MAE = {mae_cal:.2f} months)."
)
plt.figtext(0.5, 0.01, caption_text, wrap=True, ha='center', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])


os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)

plt.show()
