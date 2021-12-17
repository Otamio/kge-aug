import sys
import os
import pykeen


def get_model_mapping(dataset):

    if dataset == "fb15k237":
        from kge_aug.models.fb15k237 import Transe, Distmult, Complex, Conve, Rotate, Tucker
    elif dataset == "yago15k":
        from kge_aug.models.yago15k import Transe, Distmult, Complex, Conve, Rotate, Tucker

    return {
        "transe": Transe,
        "distmult": Distmult,
        "complex.py": Complex,
        "conve": Conve,
        "rotate": Rotate,
        "tucker": Tucker,
    }


def try_to_make_directory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def main():

    dataset = sys.argv[1]
    model = sys.argv[2]
    model_mapping = get_model_mapping(dataset)
    target_train = sys.argv[3] if len(sys.argv) > 3 else "train.tsv"
    pipeline_result = model_mapping[model.lower()].get_pipeline(dataset, target_train)
    
    if pykeen.get_version() == "1.0.0":

        metrics = pipeline_result.metric_results
        print('Model:', model)
        print('Target:', target_train)
        print('MR:', metrics.get_metric('mean_rank'))
        print('MRR:', metrics.get_metric('mean_reciprocal_rank'))
        print('Hits@1:', metrics.get_metric('hits@1'))
        print('Hits@3:', metrics.get_metric('hits@3'))
        print('Hits@5:', metrics.get_metric('hits@5'))
        print('Hits@10:', metrics.get_metric('hits@10'))

        try_to_make_directory(f"results/{dataset}")
        try_to_make_directory(f"results/{dataset}/{model}")
        try_to_make_directory(f"results/{dataset}/{model}/{target_train.split('.')[0]}")
        pipeline_result.save_to_directory(f"results/{dataset}/{model}/{target_train.split('.')[0]}")

    else:

        print('Model:', model)
        print('Target:', target_train)
        print('MR', pipeline_result.get_metric('mr'))
        print('MRR', pipeline_result.get_metric('mrr'))
        print('Hits@1', pipeline_result.get_metric('hits@1'))
        print('Hits@3', pipeline_result.get_metric('hits@3'))
        print('Hits@5', pipeline_result.get_metric('hits@5'))
        print('Hits@10', pipeline_result.get_metric('hits@10'))

        try_to_make_directory(f"results160/{dataset}")
        try_to_make_directory(f"results160/{dataset}/{model}")
        try_to_make_directory(f"results160/{dataset}/{model}/{target_train.split('.')[0]}")
        pipeline_result.save_to_directory(f"results160/{dataset}/{model}/{target_train.split('.')[0]}")


if __name__ == '__main__':
    main()
