import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class OntologyDataset(Dataset):
    def __init__(self, filepath, window_size=2):
        self.window_size = window_size
        self.sentences = []
        self.word2idx = {}
        self.idx2word = []
        self.pairs = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    self.sentences.append(tokens)

        self._build_vocab()
        self._generate_pairs()

    def _build_vocab(self):
        vocab = set(token for sentence in self.sentences for token in sentence)
        self.idx2word = list(vocab)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def _generate_pairs(self):
        for sentence in self.sentences:
            indexed = [self.word2idx[w] for w in sentence]
            for i, center in enumerate(indexed):
                for j in range(max(0, i - self.window_size), min(len(indexed), i + self.window_size + 1)):
                    if i != j:
                        context = indexed[j]
                        self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        c_emb = self.in_embed(center)
        o_emb = self.out_embed(context)
        score = torch.mul(c_emb, o_emb).sum(dim=1)
        return score

def train_word2vec(structure_file, embed_dim, window_size, batch_size, epochs, lr, use_gpu, min_delta=0.01, patience=3):
    dataset = OntologyDataset(structure_file, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    model = Word2Vec(len(dataset.word2idx), embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            center = center.to(device)
            context = context.to(device)
            label = torch.ones(center.size(0), device=device)
            optimizer.zero_grad()
            scores = model(center, context)
            loss = loss_fn(scores, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Early Stopping 判断
        if best_loss - total_loss > min_delta:
            best_loss = total_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': dataset.word2idx,
        'idx2word': dataset.idx2word
    }, 'Data/embedding/structure_embedding.pt')

    print("Saved as ./Data/structure_embedding.pt")


if __name__ == "__main__":
    train_word2vec('Data/embedding/structure.txt', embed_dim=256, window_size=2, batch_size=64, epochs=50, lr=0.001, use_gpu=True)

