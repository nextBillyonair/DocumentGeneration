DATASET = 'data_large.txt'
with open(DATASET, 'r') as content_file:
    content = content_file.read().replace('\n', ' ').replace('\t', ' ').lower()
    print(content)
