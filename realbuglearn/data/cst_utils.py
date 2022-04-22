import os
import libcst as cst

def check_indent_error(e):
    return hasattr(e, 'message') and hasattr(e, 'raw_line') and e.message.startswith('Incomplete input. Encountered an indent')

def fix_indent(e, code):
    lines = code.split('\n')
    error_line = lines[e.raw_line-1]
    indent = ''
    for c in error_line:
        if not c.isspace(): break
        indent += c

    return code.replace('\n'+indent, '\n')

def parse_with_indent_err(code):
    try:
        cst.parse_module(code)
        return code
    except Exception as e:
        if check_indent_error(e):
            code = fix_indent(e, code)
            try:
                cst.parse_module(code)
                return code
            except:
                return None
        else:
            return None

def get_info(s, task):
    s_orig = s
    items = s_orig.split(' ')
    user, repo, *path = items[1].split('/')

    func = items[2]
    func = func[func.rfind('.')+1:]

    path = '/'.join(path[2:])

    return user, repo, path, func

def iter_lines_in_dir(data_dir):
    for fname in sorted(os.listdir(data_dir)):
        if 'jsontxt' not in fname: continue

        with open(os.path.join(data_dir, fname)) as f:
            for line in f.readlines():
                yield line

def side_by_side(strings, size=95, space=3):
    ss = list(map(lambda s: s.split('\n'), strings))
    max_len = max(map(lambda s: len(s), ss))
    result = ['' for _ in range(max_len)]

    for i in range(max_len):
        for j, s in enumerate(ss):
            if i >= len(s):
                result[i] += ' ' * size
            else:
                if len(s[i]) >= size:
                    result[i] += s[i][:size]
                else:
                    result[i] += s[i] + ' ' * (size - len(s[i]))

            if j < len(ss) - 1:
                result[i] += ' ' * space + '|' + ' ' * space

    for i in range(max_len):
        result[i] = result[i].replace('\t', ' ')

    return '\n'.join(result)

def get_func_src(src, start_line, end_line):
    src_lines = src.split('\n')
    start_line -= 1
    while True:
        if start_line == 0:
            break
        if not src_lines[start_line-1].strip().startswith('@'):
            break
        start_line -= 1

    indent = ''
    for c in src_lines[start_line]:
        if not c.isspace(): break
        indent += c

    src_lines = src_lines[start_line:end_line]
    src_lines = list(map(lambda l: l[len(indent):], src_lines))
    return '\n'.join(src_lines).strip()