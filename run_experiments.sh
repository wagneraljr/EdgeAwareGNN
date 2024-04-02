MODELS=(
    AttEdgeAware
    # Não configurado ainda
    # AttEdgeAwareV2
    GCN
    GraphSAGE
)

PERIODS=(
    day
    week
)

for ((i=0; i < ${#PERIODS[@]}; i++)) do
    for ((j=0; j < ${#MODELS[@]}; j++)) do
        echo "Training ${MODELS[j]} - Task: Abilene-${PERIODS[i]}"
        python3 train.py ${MODELS[j]} ${PERIODS[i]}
    done
    python3 generate_graphics.py ${PERIODS[i]}
done