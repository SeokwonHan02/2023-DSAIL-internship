import torch

def train(args, train_dataloader, train_dataset, test_dataloader, test_dataset, model, criterion, optimizer, device):
        for epoch in range(args.epoch_num):
            model.train()
            total_loss = 0.0
            for batch_user_ids, batch_item_ids, batch_ratings in train_dataloader:
                batch_user_ids, batch_item_ids, batch_ratings = batch_user_ids.to(device), batch_item_ids.to(device), batch_ratings.to(device)

                optimizer.zero_grad()
                predictions = model(batch_user_ids, batch_item_ids)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_user_ids)
            print(f"Epoch {epoch + 1}/{args.epoch_num}, Train Loss: {total_loss / len(train_dataset)}")

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_user_ids, batch_item_ids, batch_ratings in test_dataloader:
                batch_user_ids, batch_item_ids, batch_ratings = batch_user_ids.to(device), batch_item_ids.to(device), batch_ratings.to(device)

                predictions = model(batch_user_ids, batch_item_ids)
                loss = criterion(predictions, batch_ratings)
                total_loss += loss.item() * len(batch_user_ids)
        print(f"Test Loss: {total_loss / len(test_dataset)}")