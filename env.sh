# このプロジェクト専用の環境(ほかのプロジェクトの ~/.libero とは混ぜない)
export PAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
export LIBERO_CONFIG_PATH="$PAI_ROOT/.libero"
export MUJOCO_GL=glfw PYTHONUNBUFFERED=1
export HF_HOME="$PAI_ROOT/.hf"   # モデル・アセットはこのプロジェクトの中に置く(消すときはディレクトリごと)

[ -f "$PAI_ROOT/env.local.sh" ] && source "$PAI_ROOT/env.local.sh"   # この Mac だけの設定(鍵・先生の環境の置き場、git に入れない)
