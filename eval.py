import os
import json
import copy

import torch
import numpy as np

import utils
from common import get_dataloader, maybe_dictionarize

import datasets

def eval_single_dataset(image_classifier, dataset, args, is_train=False):
    # if args.freeze_encoder:
    #     model = image_classifier.classification_head
    #     input_key = 'features'
    #     image_enc = image_classifier.image_encoder
    # else:
    print('Evaluating ...')
    model = image_classifier
    input_key = 'images'
    # image_enc = None
    model.eval()


    dataloader = get_dataloader(
        dataset, is_train=is_train, args=args)
    batched_data = enumerate(dataloader)
    device = args.device

    # if hasattr(dataset, 'post_loop_metrics'):
    #     # keep track of labels, predictions and metadata
    #     all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            logits = utils.get_logits(x, model)
            # projection_fn = getattr(dataset, 'project_logits', None)
            # if projection_fn is not None:
            #     logits = projection_fn(logits, device)

            # if hasattr(dataset, 'project_labels'):
            #     y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

        top1 = correct / n

       
    return top1

def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        # print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size
        )

        results = eval_single_dataset(image_classifier, dataset, args, is_train=False)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val
        
        json_info = copy.deepcopy(info)
        del json_info['tokenizer']
        
    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(json_info) + '\n')
        # print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

def evaluate_train(image_classifier, dataset, args):

    top1 = eval_single_dataset(image_classifier, dataset, args, is_train=False)

    print(f"  - Val set Top-1 accuracy: {top1:.4f}")
    # for key, val in results.items():
    #     info["Train set" + ':' + key] = val

    return top1