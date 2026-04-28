from dataloader import get_data_split, CandidateRankDataset

data = get_data_split("test_website", max_examples=50)
print("formatted samples:", len(data))
print("first sample keys:", data[0].keys())
print("num pos:", len(data[0]["pos_candidates"]))
print("num neg:", len(data[0]["neg_candidates"]))

dataset = CandidateRankDataset(data, neg_ratio=5)
print("dataset len:", len(dataset))
print(dataset[0])