
for service_name in qwen3-biogen lsr-biogen;do
    for field in topic question narrative; do
        python get_biogen_run.py \
            --service_endpoint 10.162.95.158:5000 \
            --service_name $service_name \
            --limit 1000 \
            --fields $field \
            --output_dir results
    done

    python get_biogen_run.py \
        --service_endpoint 10.162.95.158:5000 \
        --service_name $service_name \
        --limit 1000 \
        --fields topic \
        --fields question \
        --output_dir results
done
