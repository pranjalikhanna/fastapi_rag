import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"


from llama_cpp import Llama

def load_lamma_cpp(model_args):
    return Llama(
        model_path=model_args['model_path'],
        n_ctx=model_args['n_ctx'],
        n_batch=model_args['n_batch'],
        n_gpu_layers=model_args['n_gpu_layers']
    )
