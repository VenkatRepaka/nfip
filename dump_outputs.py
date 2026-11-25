import json, sys
nb = json.load(open(sys.argv[1]))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    for output in cell.get('outputs', []):
        text = output.get('text') or output.get('data', {}).get('text/plain')
        if output.get('output_type') == 'error':
            text = output.get('ename', '') + ': ' + output.get('evalue', '')
        if text:
            print(f'[cell {index}]', ''.join(text)[:limit])
