nohup python initialize_linker.py \
    --linker_type faiss \
    --drop_collection True \
    --cache_vector_df True \
    --release_after_initialization False \
    --dictionary_type csv \
    --corpus_type Disease \
    --extra_out_fields id \
    > log2.log 2>&1 &
