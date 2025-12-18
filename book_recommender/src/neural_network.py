import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_books, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_ids, book_ids):
        user_embed = self.user_embedding(user_ids)
        book_embed = self.book_embedding(book_ids)
        x = torch.cat([user_embed, book_embed], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)


class BookDataset(Dataset):
    def __init__(self, user_ids, book_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.book_ids = torch.tensor(book_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.book_ids[idx], self.ratings[idx]


def train_model(model, train_loader, epochs=5, progress_callback=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_ids, book_ids, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, book_ids).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if progress_callback:
            progress_callback(epoch + 1, epochs)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model
