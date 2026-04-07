# TODO — Aligner TestRetargeter sur GMR

Objectif : `TestRetargeter` produit des qpos identiques à GMR pour un même fichier SMPLX
en utilisant uniquement mink/mujoco, sans dépendance à la librairie GMR externe.

Les étapes sont ordonnées par impact décroissant sur la qualité du résultat.

---

## 1. Enrichir le pipeline pour transmettre les rotations SMPLX

**Fichiers :** `pipeline/data_loading.py`, `pipeline/run.py`

**Contexte :**  
GMR utilise les rotations absolues par joint calculées via SMPLX forward kinematics.
Le pipeline holosoma ne retourne actuellement que les positions `(T, J, 3)`.

**À faire :**
- Dans `load_motion_data`, pour `data_format == "smplx"` (et les formats AMASS),
  calculer les rotations absolues par joint via le body model SMPLX :
  ```python
  # FK: rotation absolue = composition depuis la racine
  joint_rotations[i] = joint_rotations[parents[i]] * R.from_rotvec(pose_body[i])
  ```
- Retourner un tuple optionnel `(human_joints_pos, human_joints_rot)` où
  `human_joints_rot` est `(T, J, 4)` en wxyz.
- Propager `human_joints_rot` dans `run_retargeting()` et le passer à
  `retarget_motion()` (nouveau paramètre optionnel dans `BaseRetargeter`).

**Critère de done :** `TestRetargeter._set_frame_targets()` reçoit les quaternions
et peut construire des targets `SE(3)` complets (position + orientation).

---

## 2. Implémenter les targets SE(3) complets dans _set_frame_targets()

**Fichier :** `retargeters/test.py`

**Contexte :**  
Actuellement `orientation_cost=0.0` et rotation identité. GMR utilise
`orientation_cost=10` avec les rotations SMPLX réelles + offsets de convention.

**À faire :**
- Passer `human_joints_rot: np.ndarray | None` à `_set_frame_targets()`.
- Quand disponible, construire `mink.SE3.from_rotation_and_translation(mink.SO3(rot_wxyz), pos)`.
- Activer `orientation_cost` dans la config (valeur initiale : 10, comme GMR).
- Appliquer les rotation offsets par body (voir point 4).

---

## 3. Implémenter le scaling segmentaire non-uniforme

**Fichier :** `retargeters/test.py`, `config_types/retargeters/test.py`

**Contexte :**  
GMR scale chaque segment dans le repère local du pelvis avec des facteurs différents
(jambes × 0.9, bras × 0.8). Le pipeline holosoma applique un scale uniforme global.

**À faire :**
- Ajouter un `scale_table: dict[str, float]` dans `TestRetargeterConfig`
  (valeur par défaut : identique à `smplx_to_g1.json`).
- Dans `_set_frame_targets()`, calculer la position scalée en local frame :
  ```python
  root_pos = joint_positions[demo_joints.index(root_joint)]
  local = (pos - root_pos) * scale_table[human_name]
  scaled_pos = root_pos + local
  ```
- Appliquer après la conversion en repère robot (voir point 4).

---

## 4. Aligner le mapping humain→robot sur la table GMR

**Fichier :** `retargeters/test.py` (`from_config`)

**Contexte :**  
GMR utilise `ik_match_table` (JSON) qui cible des bodies spécifiques
(`pelvis`, `left_toe_link`, `left_hip_roll_link`, ...) avec des poids
position/orientation par body.
`JOINTS_MAPPING` de holosoma cible des bodies différents
(`pelvis_contour_link`, `left_ankle_roll_sphere_5_link`, ...).

**À faire :**
- Définir dans `TestRetargeterConfig` (ou dans un fichier JSON dédié)
  la table de mapping `{robot_body_name: (human_body_name, w_pos, w_rot, pos_offset, rot_offset)}`
  calquée sur `smplx_to_g1.json`.
- Lire cette table dans `from_config()` pour créer les `FrameTask`.
- Exposer un paramètre `ik_config_path: str | None = None` pour permettre
  de pointer vers un JSON externe (compatible avec les configs GMR existantes).

---

## 5. Implémenter la résolution en deux passes (table1 → table2)

**Fichier :** `retargeters/test.py`

**Contexte :**  
GMR résout deux QPs séquentiels par frame :
- **Passe 1** (`ik_match_table1`) : ajuste l'orientation globale
  (pos_weight=0 sauf pelvis+pieds, rot_weight=10 partout).
- **Passe 2** (`ik_match_table2`) : affine les positions complètes
  (pos_weight=10 pour tous, pos_weight=100 pieds, rot_weight élevé pieds).

**À faire :**
- Créer `self._tasks_pass1` et `self._tasks_pass2` dans `from_config()`.
- Surcharger `_solve_with_refinement()` :
  ```python
  def _solve_with_refinement(self, dt):
      self._run_pass(self._tasks_pass1, dt)  # orientation globale
      self._run_pass(self._tasks_pass2, dt)  # positions complètes
  ```
- `_run_pass()` applique le même critère de convergence adaptatif que GMR
  (`while error_prev - error_next > 0.001 and iter < max_iter`).

---

## 6. Appliquer les rotation offsets par body

**Fichier :** `retargeters/test.py`

**Contexte :**  
GMR applique une rotation fixe `R_offset` à chaque body pour aligner les
conventions d'axes humain→robot (ex. pelvis : `[0.5, -0.5, -0.5, -0.5]`).
Sans cela, les orientations cibles seront dans le mauvais repère.

**À faire :**
- Lire `rot_offset` (wxyz) et `pos_offset` depuis la table de mapping (point 4).
- Dans `_set_frame_targets()` :
  ```python
  # Rotation offset
  rot_target = R.from_quat(human_rot, scalar_first=True) * R.from_quat(rot_offset, scalar_first=True)
  # Position offset (appliqué dans le repère du body après rotation)
  pos_target = pos + rot_target.apply(pos_offset)
  ```

---

## 7. Aligner le preprocessing (ground offset et hauteur sol)

**Fichier :** `retargeters/test.py` ou `pipeline/preprocessing.py`

**Contexte :**  
GMR applique un `ground_offset` configuré par robot via `ik_config["ground_height"]`
et peut ajuster la hauteur de référence au sol via `offset_human_data_to_ground()`.
Le pipeline holosoma normalise le sol via le z_min des orteils, ce qui est proche
mais pas identique.

**À faire :**
- Exposer un `ground_height: float = 0.0` dans `TestRetargeterConfig`.
- Dans `from_config()`, soustraire `ground_height * [0,0,1]` aux pos_offsets
  (même logique que `self.ground = ik_config["ground_height"] * np.array([0,0,1])`).

---

## Ordre d'implémentation recommandé

```
4 (mapping)  →  5 (deux passes)  →  6 (rot offsets)  →  3 (scaling)
                                                              ↓
                                          1 (rotations pipeline) → 2 (SE3 targets)
```

Les étapes 4–6 sont suffisantes pour avoir un comportement proche de GMR
même sans les rotations SMPLX (position-only IK reste valide pour robot_only).
Les étapes 1–2 sont nécessaires pour une équivalence exacte.

---

## Test de validation

Pour vérifier l'équivalence finale :

```bash
# Générer avec GMR (externe)
python /path/to/GMR/scripts/smplx_to_robot.py \
    --smplx_file data/seq.npz --robot unitree_g1 --save_path /tmp/gmr_out.pkl

# Générer avec TestRetargeter
python examples/robot_retarget.py \
    --retargeter-method test \
    --task-type robot_only --data_format smplx \
    --task-name seq --data_path data/

# Comparer qpos frame par frame
python -c "
import pickle, numpy as np
gmr = pickle.load(open('/tmp/gmr_out.pkl','rb'))
test = np.load('results/test/seq/retargeted.npz')
diff = np.abs(gmr['dof_pos'] - test['qpos'][:, 7:]).mean()
print(f'MAE dof_pos: {diff:.6f} rad')
"
```

Critère : MAE < 0.01 rad sur les DOF articulaires.
