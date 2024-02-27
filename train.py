import torch

def AUC(prediction_1, prediction_2):
        num_samples = len(prediction_1)
        num_correct = torch.sum(prediction_1 > prediction_2).item()

        auc = num_correct / num_samples

        return auc

def train(args, model, train_dataloader, device, optimizer):
    for epoch in range(args.epoch_num):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            user, rated_item, unrated_item = batch

            user = user.int()
            rated_item = rated_item.int()
            unrated_item = unrated_item.int()

            user, rated_item, unrated_item = user.to(device), rated_item.to(device), unrated_item.to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(user, rated_item, unrated_item)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epoch_num}, Train Loss: {total_loss}")

def evaluate(model, test_dataloader, device):
    model.eval()
    total_auc = 0.0
    num_batches = len(test_dataloader)

    with torch.no_grad():
        for batch in test_dataloader:
            user, rated_item, unrated_item = batch

            user = user.int()
            rated_item = rated_item.int()
            unrated_item = unrated_item.int()

            user, rated_item, unrated_item = user.to(device), rated_item.to(device), unrated_item.to(device)

            prediction_1 = model(user, rated_item)
            prediction_2 = model(user, unrated_item)

            auc_batch = AUC(prediction_1, prediction_2)
            total_auc += auc_batch

    avg_auc = total_auc / num_batches
    print("Average AUC:", avg_auc)