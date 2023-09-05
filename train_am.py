import os
import copy
import time
import tqdm

import numpy as np
import torch
from torch import autocast
from args import parse_arguments
from models import AudioClassifierAM
from utils import cosine_lr, torch_load, LabelSmoothing, AverageMeter, get_lr
from torch.utils.data import Dataset, DataLoader
from dataloader import AudioDataset
from losses import CosFace, ArcFace

def evaluate_train(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct, n = 0., 0.
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            logits = model(inputs).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            pred = np.argmax(logits, axis = 1)
            labels = np.squeeze(labels)
            tp = np.sum(np.where(pred == labels, 1, 0))

            correct += tp
            n += len(inputs)
    
    return correct/n

def calc_accuracy(logits, labels):
    pred = np.argmax(logits, axis = 1)
    pred = np.squeeze(pred)
    labels = np.squeeze(labels)
    assert len(pred) == len(labels)
    tp = np.sum(np.where(pred == labels, 1, 0))

    return float(tp) / float(len(pred))

def finetune(args, is_load = False):
    assert args.save is not None, "args.save must be define for saving checkpoint"
    # assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    # assert args.train_dataset is not None, "Please provide a training dataset."
    use_cuda = True if args.device == "cuda" else False

    margin_softmax = CosFace(s=64.0, m=0.4)
    model = AudioClassifierAM(arch = 'resnet50', num_classes = args.total_classes, margin_softmax = margin_softmax)
    if args.load is not None:
        print('  - Loading ', args.load)
        model = model.load(args.load)

    print(model)
    print('  - Init train dataloader')
    train_dataset = AudioDataset(
        root = args.train_txt,
        total_classes = args.total_classes,
    )
    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )


    val_dataset = AudioDataset(
        root = args.val_txt,
        total_classes = args.total_classes,
        is_train = False
    )
    val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

    num_batches = len(train_loader)
    print('    + Number of batches per epoch: {}'.format(num_batches))

    if use_cuda:
        model = model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print('  - Using device_ids: ', devices)
        # model = torch.nn.DataParallel(model, device_ids=devices)


    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
        print('  - Init LabelSmoothingLoss')
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print('  - Init CrossEntropyLoss')

    # if args.freeze_encoder:
    #     print('  - Freeze backbone')
    #     model.module.image_encoder.model.requires_grad_(False)
    #     model.module.classification_head.requires_grad_(True)
    # else:
    #     model.module.image_encoder.model.requires_grad_(True)
    #     model.module.classification_head.requires_grad_(True)

    params      = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters() if p.requires_grad]
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    print('  - Init AdamW with cosine learning rate scheduler')

    if args.fp16:
        print('  - Using Auto mixed precision')
        scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    start_epoch = 0
    best_acc = 0.0
    print_every = 100
    for epoch in range(start_epoch, args.epochs):
        print(f"Start epoch: {epoch}")
        model.train()
    
        # if args.freeze_encoder:
        #     print('  - Freeze backbone')
        #     model.module.image_encoder.model.requires_grad_(False)
        #     model.module.classification_head.requires_grad_(True)
        # else:
        #     model.module.image_encoder.model.requires_grad_(True)
        #     model.module.classification_head.requires_grad_(True)
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        

        for i, (inputs, labels) in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            lr = scheduler(step)
            optimizer.zero_grad()

            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            data_time.update(time.time() - start_time)
            # data_time = time.time() - start_time
            start_time = time.time()
            # compute output
            if args.fp16:
                with autocast(device_type="cuda", dtype=torch.float16):
                    features = model.forward_backbone(inputs)
                    loss = model.calculate_am_loss(features, labels)
                losses.update(loss.item(), inputs.size(0))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.zero_grad()
            else:
                features = model.forward_backbone(inputs)
                loss = model.calculate_am_loss(features, labels)

                losses.update(loss.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time.update(time.time() - start_time)
            # start_time = time.time()

            if i > 0 and i % print_every == 0:
                logits_am_np = model.forward_am(inputs, labels).detach().cpu().numpy()
                logits_np = model(inputs).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                train_am_acc = calc_accuracy(logits_am_np, labels_np)
                train_acc = calc_accuracy(logits_np, labels_np)
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t",
                    "Lr: {}\tLoss: {}\tAM-Accuracy: {} \tAccuracy: {}\tData (t) {}\tBatch (t) {}".format(get_lr(optimizer), loss.item(), train_am_acc, train_acc, data_time.avg, batch_time.avg), 
                    flush=True
                )

            if i > 0 and i % args.valid_every == 0:
                print('Valid iters ', i)
                # Evaluate
                args.current_epoch = epoch
                args.current_loss = losses.avg

                tik = time.time()
                val_acc = evaluate_train(model, val_loader)
                tok = time.time()
                print('Eval done in', tok - tik, "  Accuracy = ", val_acc)
                is_best = val_acc >= best_acc

                # Saving model
                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    if is_best:
                        print('  - Saving as best checkpoint')
                        model.save(os.path.join(args.save, f'checkpoint_model_best.pt'))
                        best_acc = val_acc
                model.train()



        print(f"Epoch {epoch}:\t Loss: {losses.avg:.5f}\t"
              f"Data(t): {data_time.avg:.3f}\t Batch(t): {batch_time.avg:.3f}")

        # if args.freeze_encoder:
        #     image_classifier = ImageClassifier(image_classifier.image_encoder, model.module) if use_cuda \
        #     else ImageClassifier(image_classifier.image_encoder, model)
        # else:
        # image_classifier = model.module if use_cuda else model

        # Evaluate
        # args.current_epoch = epoch
        # args.current_loss = losses.avg

        # tik = time.time()
        # val_acc = evaluate_train(image_classifier, dataset, args)
        # tok = time.time()
        # print('Eval done in', tok - tik)
        # is_best = val_acc > best_acc

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            if epoch % args.save_interval == 0:
                model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
                model.save(model_path)
              
            # if is_best:
            #     print('  - Saving as best checkpoint')
            #     image_classifier.save(os.path.join(args.save, f'checkpoint_model_best.pt'))
            #     best_acc = val_acc
            
    if args.save is not None:
        return model_path

if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)