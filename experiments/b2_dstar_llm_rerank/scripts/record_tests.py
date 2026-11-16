import xml.etree.ElementTree as ET
from common_io import ROOT,dump,sha,now
root=ET.parse(ROOT/'results/tests.xml').getroot()
suites=list(root.iter('testsuite'))
tests=sum(int(s.get('tests',0)) for s in suites)
failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
assert tests>=19 and failures==0
dump(ROOT/'results/tests.json',dict(passed=True,tests=tests,failures=failures,at=now(),
    source_sha256=sha(ROOT/'tests/test_rerank_isolation.py'),xml_sha256=sha(ROOT/'results/tests.xml')))
