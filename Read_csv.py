import numpy as np
import seaborn as sns
import pygmo as pg
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from pyDOE import lhs
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from nds import ndomsort
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

def process_file(filepath):
    columns = [
        "Input 1 (Ego Speed)", "Input 2 (Target Speed)",
        "Output 1 (TTC)", "Output 2 (Jerk)",
        "HyperVolume", "Generational Distance",
        "Is_Pareto"
    ]
    df = pd.read_csv(filepath, header=None, skiprows=2, names=columns)

    return df

n_iterations = 200 
resolution = 0.5

df = process_file(f'Results_{n_iterations}_iterations.csv')
df_sweeping = process_file(f'Results_Sweeping_{resolution}_Resolution.csv')

AEB_scenario_final1_y_vals = df["Output 1 (TTC)"]
AEB_scenario_final2_y_vals = df["Output 2 (Jerk)"]
AEB_scenario_final1_lhs_samples = df["Input 1 (Ego Speed)"]
AEB_scenario_final2_lhs_samples = df["Input 2 (Target Speed)"]

pareto_front = df[df["Is_Pareto"] == "Pareto"]
AEB_scenario_pareto_front = pareto_front[["Output 1 (TTC)", "Output 2 (Jerk)"]].to_numpy()
AEB_scenario_pareto_inputs = pareto_front[["Input 1 (Ego Speed)", "Input 2 (Target Speed)"]].to_numpy()

Sweeping_y1_vals = df_sweeping["Output 1 (TTC)"]
Sweeping_y2_vals = df_sweeping["Output 2 (Jerk)"]
Sweeping_x1_vals = df_sweeping["Input 1 (Ego Speed)"]
Sweeping_x2_vals = df_sweeping["Input 2 (Target Speed)"]

pf_sweeping = df_sweeping[df_sweeping["Is_Pareto"] == "Pareto"]
pareto_front_sweeping = pf_sweeping[["Output 1 (TTC)", "Output 2 (Jerk)"]].to_numpy()
inputs_pareto_sweeping = pf_sweeping[["Input 1 (Ego Speed)", "Input 2 (Target Speed)"]].to_numpy()

min_ttc_AEB, max_ttc_AEB = np.min(Sweeping_y1_vals), np.max(Sweeping_y1_vals)
min_jerk_AEB, max_jerk_AEB = np.min(Sweeping_y2_vals), np.max(Sweeping_y2_vals)

ttc_bounds_AEB = [min_ttc_AEB-0.05, 1.0, 1.4, max_ttc_AEB+0.05]
jerk_bounds_AEB = [min_jerk_AEB-1.0, -10.0, -5.0, max_jerk_AEB+1.0]

red_points_bayes = []         
lightcoral_points_bayes = []  
green_points_bayes = []       
yellow_points_bayes = []      

red_idx_bayes = []
lightcoral_idx_bayes = []
green_idx_bayes = []
yellow_idx_bayes = []

for idx, (ttc, jerk) in enumerate(zip(AEB_scenario_final1_y_vals, AEB_scenario_final2_y_vals)):
    if ttc <= ttc_bounds_AEB[1] and jerk <= jerk_bounds_AEB[1]:
        red_points_bayes.append((ttc, jerk))
        red_idx_bayes.append(idx)
    
    elif (ttc > ttc_bounds_AEB[1] and jerk <= jerk_bounds_AEB[1]) or \
        (ttc <= ttc_bounds_AEB[1] and jerk > jerk_bounds_AEB[1]):
        lightcoral_points_bayes.append((ttc, jerk))
        lightcoral_idx_bayes.append(idx)
    
    elif ttc > ttc_bounds_AEB[1] and ttc <= ttc_bounds_AEB[2] and \
        jerk > jerk_bounds_AEB[1] and jerk <= jerk_bounds_AEB[2]:
        green_points_bayes.append((ttc, jerk))
        green_idx_bayes.append(idx)
    
    else:
        yellow_points_bayes.append((ttc, jerk))
        yellow_idx_bayes.append(idx)

red_points_sweep = []         
lightcoral_points_sweep = []  
green_points_sweep = []       
yellow_points_sweep = []      

red_idx_sweep = []
lightcoral_idx_sweep = []
green_idx_sweep = []
yellow_idx_sweep = []

for idx, (ttc, jerk) in enumerate(zip(Sweeping_y1_vals, Sweeping_y2_vals)):
    if ttc <= ttc_bounds_AEB[1] and jerk <= jerk_bounds_AEB[1]:
        red_points_sweep.append((ttc, jerk))
        red_idx_sweep.append(idx)
    
    elif (ttc > ttc_bounds_AEB[1] and jerk <= jerk_bounds_AEB[1]) or \
        (ttc <= ttc_bounds_AEB[1] and jerk > jerk_bounds_AEB[1]):
        lightcoral_points_sweep.append((ttc, jerk))
        lightcoral_idx_sweep.append(idx)
    
    elif ttc > ttc_bounds_AEB[1] and ttc <= ttc_bounds_AEB[2] and \
        jerk > jerk_bounds_AEB[1] and jerk <= jerk_bounds_AEB[2]:
        green_points_sweep.append((ttc, jerk))
        green_idx_sweep.append(idx)
    
    else:
        yellow_points_sweep.append((ttc, jerk))
        yellow_idx_sweep.append(idx)

        
# Objective Space with Colors (Sweeping)

plt.close('all')
plt.figure(figsize=(7, 5))

plt.fill_between(
    [ttc_bounds_AEB[0], ttc_bounds_AEB[1]], 
    jerk_bounds_AEB[0], jerk_bounds_AEB[1], 
    color='red', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[0], jerk_bounds_AEB[1], 
    color='lightcoral', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[0], ttc_bounds_AEB[1]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[3], 
    color='lightcoral', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[2]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[2], 
    color='green', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[2], jerk_bounds_AEB[3], 
    color='yellow', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[2], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[2], 
    color='yellow', alpha=0.3
)

plt.scatter(Sweeping_y1_vals, Sweeping_y2_vals, edgecolors='black', facecolors='none', s=10)
plt.scatter(pareto_front_sweeping[:, 0], pareto_front_sweeping[:, 1], color='green', marker='*', s=30) 
pareto_labels_sweep_AEB = [f"{i+1}" for i in range(len(pareto_front_sweeping))]
for i, label in enumerate(pareto_labels_sweep_AEB):
    plt.annotate(label,(pareto_front_sweeping[i, 0], pareto_front_sweeping[i, 1]), textcoords="offset points",  xytext=(-10, -10),  ha='left',fontsize=9,color='black',arrowprops=dict(arrowstyle="->",lw=1,color='black'))
#plt.title('Objective Space of Sweeping Method')
plt.xlabel('Time-to-Collision (s)', fontsize = 17)
plt.ylabel('Jerk (m/s³)', fontsize = 17)
plt.legend(loc='upper right', frameon=False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
plt.savefig(f"Plots/Objective_Space_with_Colors_(Sweeping).pdf", bbox_inches='tight', dpi=600)

# Design Space With Colors (Sweeping)
    
plt.close('all')
plt.figure(figsize=(7, 5))

if red_idx_sweep:
    for l in red_idx_sweep:
        point = (Sweeping_x1_vals[l], Sweeping_x2_vals[l])
        facecolor, marker = 'red', 'o'
        plt.scatter(Sweeping_x1_vals[l], Sweeping_x2_vals[l], 
                    facecolor=facecolor, s=5,
                    edgecolors='red', alpha=1.0, marker=marker, linewidths=2)
        
if lightcoral_idx_sweep:
    for l in lightcoral_idx_sweep:
        point = (Sweeping_x1_vals[l], Sweeping_x2_vals[l])
        facecolor, marker = 'lightcoral', 'o'
        plt.scatter(Sweeping_x1_vals[l], Sweeping_x2_vals[l], 
                    facecolor=facecolor, s=5,
                    edgecolors='lightcoral', alpha=1.0, marker=marker, linewidths=2)

if green_idx_sweep:
    for l in green_idx_sweep:
        point = (Sweeping_x1_vals[l], Sweeping_x2_vals[l])
        facecolor, marker = 'green', 'o'
        plt.scatter(Sweeping_x1_vals[l], Sweeping_x2_vals[l], 
                    facecolor=facecolor, s=5,
                    edgecolors='green', alpha=1.0, marker=marker, linewidths=2)
        
if yellow_idx_sweep:
    for l in yellow_idx_sweep:
        point = (Sweeping_x1_vals[l], Sweeping_x2_vals[l])
        facecolor, marker = 'yellow', 'o'
        plt.scatter(Sweeping_x1_vals[l], Sweeping_x2_vals[l], 
                    facecolor=facecolor, s=5,
                    edgecolors='yellow', alpha=1.0, marker=marker, linewidths=2)

pareto_labels_sweeping_AEB = [f"{i+1}" for i in range(len(pareto_front_sweeping))]
for i, label in enumerate(pareto_labels_sweeping_AEB):
    plt.annotate(label, (inputs_pareto_sweeping[i, 0], inputs_pareto_sweeping[i, 1]), textcoords="offset points",  xytext=(10, 5),ha='left',  fontsize=9,color='black',arrowprops=dict(arrowstyle="->", lw=1, color='black'))
plt.xlabel('Ego Velocity (m/s)', fontsize = 17)
plt.ylabel('Target Velocity (m/s)', fontsize = 17)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig(f'Plots/Design_Space_With_Colors_(Sweeping).pdf', bbox_inches='tight', dpi=600)

# Objective Space with Colors (Bayesian)

plt.close('all')
plt.figure(figsize=(7, 5))

plt.fill_between(
    [ttc_bounds_AEB[0], ttc_bounds_AEB[1]], 
    jerk_bounds_AEB[0], jerk_bounds_AEB[1], 
    color='red', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[0], jerk_bounds_AEB[1], 
    color='lightcoral', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[0], ttc_bounds_AEB[1]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[3], 
    color='lightcoral', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[2]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[2], 
    color='green', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[1], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[2], jerk_bounds_AEB[3], 
    color='yellow', alpha=0.3
)

plt.fill_between(
    [ttc_bounds_AEB[2], ttc_bounds_AEB[3]], 
    jerk_bounds_AEB[1], jerk_bounds_AEB[2], 
    color='yellow', alpha=0.3
)
    
plt.scatter(AEB_scenario_final1_y_vals, AEB_scenario_final2_y_vals, facecolors='none', edgecolors='black', s=10)
plt.scatter(AEB_scenario_pareto_front[:, 0], AEB_scenario_pareto_front[:, 1], color='red', marker='x', s=30)
pareto_labels_bayes_AEB = [f"{i+1}" for i in range(len(AEB_scenario_pareto_front))]
for i, label in enumerate(pareto_labels_bayes_AEB):
    plt.annotate(label,(AEB_scenario_pareto_front[i, 0], AEB_scenario_pareto_front[i, 1]), textcoords="offset points",  xytext=(-10, -10),  ha='left',fontsize=9,color='black',arrowprops=dict(arrowstyle="->",lw=1,color='black'))
#plt.title(f'Objective Space of Bayesian Algorithm ({n_iterations} iterations)')
plt.xlabel('Time-to-Collision (s)', fontsize = 17)
plt.ylabel('Jerk (m/s³)', fontsize = 17)
plt.legend(loc='upper right', frameon=False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
plt.savefig(f"Plots/Objective_Space_with_Colors_(Bayesian).pdf", bbox_inches='tight', dpi=600)

# Design Space With Colors (Bayesian)

plt.close('all')
plt.figure(figsize=(7, 5))

if red_idx_bayes:
    for l in red_idx_bayes:
        point = (AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l])
        facecolor, marker = 'red', 'o'
        plt.scatter(AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l],
                    facecolor=facecolor,s=5,
                    edgecolors='red', alpha=1.0, marker=marker, linewidths=2)

if lightcoral_idx_bayes:
    for l in lightcoral_idx_bayes:
        point = (AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l])
        facecolor, marker = 'lightcoral', 'o'
        plt.scatter(AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l],
                    facecolor=facecolor, s=5,
                    edgecolors='lightcoral', alpha=1.0, marker=marker, linewidths=2)

        
if green_idx_bayes:
    for l in green_idx_bayes:
        point = (AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l])
        facecolor, marker = 'green', 'o'
        plt.scatter(AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l],
                    facecolor=facecolor, s=5,
                    edgecolors='green', alpha=1.0, marker=marker, linewidths=2)

if yellow_idx_bayes:
    for l in yellow_idx_bayes:
        point = (AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l])
        facecolor, marker = 'yellow', 'o'
        plt.scatter(AEB_scenario_final1_lhs_samples[l], AEB_scenario_final2_lhs_samples[l], 
                    facecolor=facecolor, s=5,
                    edgecolors='yellow', alpha=1.0, marker=marker, linewidths=2)    

for i, label in enumerate(pareto_labels_bayes_AEB):
    plt.annotate(label,(AEB_scenario_pareto_inputs[i, 0], AEB_scenario_pareto_inputs[i, 1]), textcoords="offset points",  xytext=(10, 5),ha='left',  fontsize=9,color='black',arrowprops=dict(arrowstyle="->", lw=1, color='black'))
plt.xlabel('Ego Velocity (m/s)', fontsize = 17)
plt.ylabel('Target Velocity (m/s)', fontsize = 17)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig(f'Plots/Design_Space_With_Colors_(Bayesian).pdf', bbox_inches='tight', dpi=600)

# Predicted Pareto #################################################################

def load_model( filename):
    data = joblib.load(filename)
    model = data['model']
    
    print('Model loaded successfully!')
    return model

def predict_yvalue(model, ego_vel, targe_vel):
    x_new = np.array([[ego_vel, targe_vel]])
    
    y_pred, y_std = model.predict(x_new, return_std = True)
    
    return y_pred, y_std


def save_model(filename, gp_model):    
        joblib.dump({
            'model': gp_model
        }, filename)

def sort_points_by_x(points):
    return points[np.argsort(points[:, 0])]

TTC_values = np.array(AEB_scenario_final1_y_vals)
Jerk_Values = np.array(AEB_scenario_final2_y_vals)

Ego_Speed_Values = AEB_scenario_final1_lhs_samples
Target_Speed_Values = AEB_scenario_final2_lhs_samples

x_values = np.array([[e, t] for e, t in zip(Ego_Speed_Values, Target_Speed_Values)])

kernel_jerk = RationalQuadratic(length_scale=1.0, alpha=1.0)
kernel_TTC = RationalQuadratic(length_scale=1.0, alpha=1.0)

gp_model_jerk = GaussianProcessRegressor(kernel=kernel_jerk, n_restarts_optimizer=100, normalize_y=True)
gp_model_TTC = GaussianProcessRegressor(kernel=kernel_TTC, n_restarts_optimizer=100, normalize_y=True)

gp_model_TTC.fit(x_values, TTC_values)
gp_model_jerk.fit(x_values, Jerk_Values)

save_model('Kriging_model_TTC_only_OFF.pkl', gp_model_TTC)
save_model('Kriging_model_Jerk_only_OFF.pkl', gp_model_jerk)

ego_min, ego_max = 14, 25
target_min, target_max = 0, 11

num_ego_points = int(round((ego_max - ego_min) / 0.1)) + 1
num_target_points = int(round((target_max - target_min) / 0.1)) + 1

ego_vals = np.linspace(ego_min, ego_max, num_ego_points)
target_vals = np.linspace(target_min, target_max, num_target_points)
print(ego_vals)
print(target_vals)

ego_grid, target_grid = np.meshgrid(ego_vals, target_vals)

X_scaled = np.vstack([ego_grid.ravel(), target_grid.ravel()]).T

kriging_model_1 = load_model('Kriging_model_TTC_only_OFF.pkl')
kriging_model_2 = load_model('Kriging_model_Jerk_only_OFF.pkl')
TTC_predicted = []
Jerk_predicted = []

for ego_vel, target_vel in zip(X_scaled[:, 0], X_scaled[:, 1]):
    predictions1, stds1 = predict_yvalue(kriging_model_1, ego_vel, target_vel)  
    predictions2, stds2 = predict_yvalue(kriging_model_2, ego_vel, target_vel) 
    TTC_predicted.append(predictions1.item())
    Jerk_predicted.append(predictions2.item())

y_vals_for_pf_predicted = np.column_stack((TTC_predicted, Jerk_predicted))

fronts = ndomsort.non_domin_sort(y_vals_for_pf_predicted, only_front_indices=True)

pareto_front_indices_predicted = [i for i, f in enumerate(fronts) if f == 0]

predicted_pareto_front = sort_points_by_x(np.vstack((y_vals_for_pf_predicted[pareto_front_indices_predicted], AEB_scenario_pareto_front)))

x_lim = (0.6, 1.5)
y_lim = (-14.0, -9.0)

# Pareto bayesian + pareto sweeping

plt.close('all')
plt.figure(figsize=(7, 5))
plt.plot(sort_points_by_x(AEB_scenario_pareto_front)[:, 0], sort_points_by_x(AEB_scenario_pareto_front)[:, 1], color = 'black') 
plt.scatter(AEB_scenario_pareto_front[:, 0], AEB_scenario_pareto_front[:, 1], color='red', marker='x', s=30)
plt.xlim(x_lim) 
plt.ylim(y_lim)
#plt.title(f'Pareto Front of Bayesian Algorithm ({n_iterations} iterations)')
plt.xlabel('Time-to-Collision (s)', fontsize = 17)
plt.ylabel('Jerk (m/s³)', fontsize = 17)
plt.legend(loc='upper right', frameon=False)
plt.grid(False)
plt.tight_layout()
plt.savefig(f"Plots/Pareto_fronts_AEB_{n_iterations}.pdf", bbox_inches='tight', dpi=600)

# Pareto Predicted + pareto sweeping


plt.figure(figsize=(7, 5))
plt.scatter(AEB_scenario_pareto_front[:, 0], AEB_scenario_pareto_front[:, 1], color='red', marker='x', s=40)
plt.plot(sort_points_by_x(predicted_pareto_front)[:, 0], sort_points_by_x(predicted_pareto_front)[:, 1], color = 'black')
plt.plot(sort_points_by_x(pareto_front_sweeping)[:, 0], sort_points_by_x(pareto_front_sweeping)[:, 1], color='green', alpha=0.6) 
plt.scatter(pareto_front_sweeping[:, 0], pareto_front_sweeping[:, 1], color='blue', marker='*', s=40, alpha=0.6) 
plt.xlim(x_lim) 
plt.ylim(y_lim)
plt.xlabel('Time-to-Collision (s)', fontsize = 17)
plt.ylabel('Jerk (m/s³)', fontsize = 17)
plt.legend(loc='upper right', frameon=False)
#plt.title('Predicted Pareto Front')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Plots/Pareto_Front_Predicted_vs_sweeping.pdf', bbox_inches='tight', dpi=600)

# 3D Jerk

Z_jerk_pred = np.array(Jerk_predicted).reshape(ego_grid.shape)
Z_ttc_pred = np.array(TTC_predicted).reshape(ego_grid.shape)

Z_jerk_sweep = griddata((Sweeping_x1_vals, Sweeping_x2_vals), Sweeping_y2_vals, (ego_grid, target_grid), method='linear')
Z_ttc_sweep = griddata((Sweeping_x1_vals, Sweeping_x2_vals), Sweeping_y1_vals, (ego_grid, target_grid), method='linear')

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=135)  # Ângulo da visualização

# Superfícies
surf_pred = ax.plot_surface(
    ego_grid, target_grid, Z_jerk_pred,
    cmap='Reds', alpha=0.7, edgecolor='none'
)
surf_sweep = ax.plot_surface(
    ego_grid, target_grid, Z_jerk_sweep,
    cmap='Greens', alpha=0.5, edgecolor='none'
)

# Rótulos
ax.set_xlabel('Ego Speed', fontsize=15, labelpad=10)
ax.set_ylabel('Target Speed', fontsize=15, labelpad=10)
ax.set_zlabel('Jerk', fontsize=15, labelpad=15)

# Estilo dos ticks
ax.tick_params(axis='both', labelsize=12)
ax.zaxis.set_tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('Plots/3d_Surface_Jerk.pdf', dpi=600)

# === TTC PLOT ===

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=135)  # Mesmo ângulo para consistência

surf_pred = ax.plot_surface(
    ego_grid, target_grid, Z_ttc_pred,
    cmap='Blues', alpha=0.7, edgecolor='none'
)
surf_sweep = ax.plot_surface(
    ego_grid, target_grid, Z_ttc_sweep,
    cmap='Greens', alpha=0.5, edgecolor='none'
)

ax.set_xlabel('Ego Speed', fontsize=15, labelpad=10)
ax.set_ylabel('Target Speed', fontsize=15, labelpad=10)
ax.set_zlabel('TTC', fontsize=15, labelpad=15)

ax.tick_params(axis='both', labelsize=12)
ax.zaxis.set_tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('Plots/3d_Surface_TTC.pdf', dpi=600)
