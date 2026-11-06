import json, struct, sys
def header(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))
if __name__ == '__main__':
    h = header(sys.argv[1])
    for k, v in h.items():
        if k != '__metadata__':
            print(k, v['dtype'], v['shape'], v['data_offsets'][1] - v['data_offsets'][0])
