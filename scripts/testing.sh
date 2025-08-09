## -------------- Listwise -------------- ##
## PLAIDX
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n02:5000 \
#     --service_name plaidx-listllama-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.plaidx_rankgpt
# python run_rac_nugget_eval.py \
#     --host http://rack2n05 \
#     --port 5000 \
#     --service_name plaidx-listllama-neuclir \
#     --dataset_name neuclir

## fusion3 + rankgpt
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n06:5000 \
#     --service_name fusion3-listllama-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.fusion3_rankgpt
# python run_rac_nugget_eval.py \
#     --host http://rack2n06 \
#     --port 5000 \
#     --service_name fusion3-listllama-neuclir \
#     --dataset_name neuclir

## fusion2 + rankgpt
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n03:5000 \
#     --service_name fusion2-listllama-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.fusion2_pointwisellama
# python run_rac_nugget_eval.py \
#     --host http://rack2n03 \
#     --port 5000 \
#     --service_name fusion2-listllama-neuclir \
#     --dataset_name neuclir

## -------------- Pointwise -------------- ##
## PLAIDX
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n05:5000 \
#     --service_name plaidx-pointllama-neuclir \
#     --limit 1000 --max_limit 1000 \
# python run_rac_nugget_eval.py \
#     --host http://rack2n05 \
#     --port 5000 \
#     --service_name plaidx-pointllama-neuclir \
#     --dataset_name neuclir

## fusion3 + pointiwisellama
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n02:5000 \
#     --service_name fusion3-pointllama-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.fusion3_pointwisellama
# python run_rac_nugget_eval.py \
#     --host http://rack2n02 \
#     --port 5000 \
#     --service_name fusion3-pointllama-neuclir \
#     --dataset_name neuclir

## fusion2 + pointwisellama
# python neuclir_ir_eval.py \
#     --service_endpoint rack2n05:5000 \
#     --service_name fusion2-pointllama-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.fusion2_pointwisellama
# python run_rac_nugget_eval.py \
#     --host http://rack2n05 \
#     --port 5000 \
#     --service_name fusion2-pointllama-neuclir \
#     --dataset_name neuclir

## -------------- First-stage -------------- ##
# python neuclir_ir_eval.py \
#     --service_endpoint 10.162.95.158:5000 \
#     --service_name plaidx-neuclir \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.plaidx
# python neuclir_ir_eval.py \
#     --service_endpoint 10.162.95.158:5000 \
#     --service_name 'rrf(lsr+qwen)-neuclir' \
#     --limit 1000 --max_limit 1000 \
#     --output_dir results.fusion2


# for service in qwen3-ragtime plaidx-ragtime lsr-ragtime;do
for service in fusion3-listllama-ragtime;do
    python get_ragtime_runs.py \
    --service_endpoint http://rack2n12:5000 \
    --service_name ${service} \
    --topics_path /exp/scale25/ragtime/topics/topic25_dry.jsonl \
    --output_dir ragtime_runs \
    --prefix ragtime_t+ps \
    --limit 1000
done

# for service in plaidx-pointllama-ragtime;do
# for service in qwen3-pointllama-ragtime;do
# for service in lsr-pointllama-ragtime;do
# python get_ragtime_runs.py \
#     --service_endpoint http://rack2n02:5000 \
#     --service_name ${service} \
#     --topics_path /exp/scale25/ragtime/topics/topic25_dry.jsonl \
#     --output_dir ragtime_runs \
#     --prefix ragtime_t+ps \
#     --limit 500
# done

# for service in plaidx-listllama-ragtime;do
# for service in qwen3-listllama-ragtime;do
# for service in lsr-listllama-ragtime;do
# for service in fusion2-listllama-ragtime;do
# python get_ragtime_runs.py \
#     --service_endpoint http://rack2n02:5000 \
#     --service_name ${service} \
#     --topics_path /exp/scale25/ragtime/topics/topic25_dry.jsonl \
#     --output_dir ragtime_runs \
#     --prefix ragtime_t+ps \
#     --limit 1000
# done
