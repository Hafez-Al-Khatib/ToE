import json
with open('outputs/scaled_eval/cifar10_scaled_500.json') as f:
    data = json.load(f)
print('Top keys:', list(data.keys()))
if 'results' in data:
    print('Results keys:', list(data['results'].keys()))
    for k in list(data['results'].keys()):
        print(' ', k, 'type:', type(data['results'][k]))
else:
    print('No results key')
    print('Available:', list(data.keys()))
