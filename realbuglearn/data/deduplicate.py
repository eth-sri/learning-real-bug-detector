import json
import os
import io
import keyword
from tokenize import generate_tokens
from tqdm import tqdm
import sys

def get_detector_repo():
    # environment var
    if "DEDUP_DETECTOR_REPO" in os.environ:
        return os.environ["DEDUP_DETECTOR_REPO"]
    # relative to scriopt
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "near-duplicate-code-detector")

def get_detector_cmd():
    return "dotnet run --project {}/DuplicateCodeDetector/DuplicateCodeDetector.csproj".format(get_detector_repo())

class MissingDependencyException(Exception):
    pass

def ensure_dependencies():
    if os.system("gzip --version > /dev/null 2>&1") != 0:
        raise MissingDependencyException("gzip not found")
    if os.system("dotnet --version  > /dev/null 2>&1") != 0:
        raise MissingDependencyException("dotnet not found: please install a recent version of the dotnet SDK.")
    if not os.path.exists(get_detector_repo()):
        raise MissingDependencyException("microsoft/near-duplicate-code-detector not found: please make sure a copy is located at {}, i.e.\n git clone https://github.com/microsoft/near-duplicate-code-detector.git {}".format(get_detector_repo(), get_detector_repo()))
    if os.system(get_detector_cmd() + " --help  > /dev/null 2>&1") != 0:
        raise MissingDependencyException("near-duplicate-code-detector could not be executed correctly.")

"""
Python wrapper for using the microsoft/near-duplicate-code-detector.
"""

def deduplicate(code_samples, code_attr="code", key=lambda i, s: s["key"]):
    """
    Deduplicates the given list of Python code samples using 
    the near-duplicates detection of microsoft/near-duplicate-code-detector.

    Returns an iterator of all non-near-duplicates in code_samples.

    Parameters:

    code_samples : list of {"code": str, ...} where id must be unique
    code_attr: attribute name of of the string-represented code of a sample
    key: function that returns the id/key of a code sample (must be unique).
    """
    ensure_dependencies()

    tmp_dupl_input = "__tmp_dupl_input.jsonl"
    tmp_dupl_output = "__tmp_dupl_output"

    f_out = open(tmp_dupl_input, "w")

    # transform list of code samples into required jsonl format
    for i, sample in enumerate(code_samples):
        tokens = []
        try:
            code = sample[code_attr]
            if sys.version_info.major == 2:
                if not isinstance(code, unicode): 
                    code = code.decode("utf-8")
            buffer = io.StringIO(code)
            for _, tokval, _, _, _ in generate_tokens(buffer.readline):
                if not keyword.iskeyword(tokval):
                    tokens.append(tokval)
        except Exception as e:
            print('Error tokenizing %s: %s' % (key(i, sample), str(e)))
            continue
    
        f_out.write(json.dumps({
            "filename": key(i, sample),
            "tokens": tokens,
        }) + '\n')
    
    f_out.close()

    # compress jsonl file
    os.system("gzip -f " + tmp_dupl_input)
    
    # run duplicate detecor
    print(get_detector_cmd() + " --input={} {}".format(tmp_dupl_input + ".gz", tmp_dupl_output))
    os.system(get_detector_cmd() + " --input={} {}".format(tmp_dupl_input + ".gz", tmp_dupl_output))

    # read deduplicator output
    is_duplicate = set()

    with open(tmp_dupl_output + ".json", "r") as f:
        dup_groups = json.load(f)
        for group in dup_groups:
            # mark all but first instance as duplicate
            for sample_id in list(sorted(group))[1:]:
                is_duplicate.add(sample_id)

    # filter near-duplicates from code samples
    for i, s in enumerate(code_samples):
        sample_id = key(i, s)
        if str(sample_id) not in is_duplicate:
            yield s

if __name__ == "__main__":
    print("checking dependencies for deduplicate.py...")
    ensure_dependencies()

    print("running basic near-duplicates functionality test...")
    
    # some basic functionality test
    res = deduplicate([
        {"code": u"def outer(view):\n    def inner(request, id, slug=''):\n        instance = get_object_or_404(model, pk=id)\n        if not request.path == instance.get_absolute_url():\n            return redirect(instance, permanent=True)\n        return view(request, instance)\n    return inner", "id": "foo"},
        {"code": u"def canonical(model):\n    \"\"\"\n    Enforce a canonical URL for a resource.\n    \"\"\"\n    def outer(view):\n        def inner(request, id, slug=''):\n            instance = get_object_or_404(model, pk=id)\n            if not request.path == instance.get_absolute_url():\n                return redirect(instance, permanent=True)\n            return view(request, instance)\n        return inner\n    return outer", "id": "foo2"},
        {"code": u"def bar():\n    pass", "id": "bar"},
    ], "code", key=lambda i,s: s["id"])
    res = list(res)
    ids = sorted([s["id"] for s in res])
    assert len(res) == 2, "deduplication correctly filters simple duplicate"
    assert "bar" in ids, "deduplication correctly filters simple duplicate"