import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('watch_experiment',Path(__file__).with_name('watch_experiment.py'))
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)

def decision(alive=False,status=None,complete=False,free=100*1024**3,quiet=0,restarts=0,changed=None):
    return watch.classify(alive,status or {},complete,free,quiet,restarts,2,changed or [])

def test_live_process_is_never_restarted():
    assert decision(alive=True)=='healthy'
    assert decision(alive=True,quiet=3600)=='quiet'

def test_unexpected_exit_can_resume():
    assert decision(status={'state':'running'})=='restart'

def test_cuda_oom_can_resume():
    assert decision(status={'state':'failed','error':'CUDA out of memory'})=='restart'

@pytest.mark.parametrize('error',['AssertionError: hidden inverse','TypeError: bad object','ValueError: signature mismatch','protected historical assets changed'])
def test_deterministic_failure_is_not_retried(error):
    assert decision(status={'state':'failed','traceback':error})=='needs_attention_deterministic_failure'

def test_disk_and_source_changes_stop_recovery():
    assert decision(free=1024)=='needs_attention_disk'
    assert decision(changed=['src/path_extractor.py'])=='needs_attention_source_changed'

def test_unknown_errors_require_attention():
    assert decision(status={'state':'failed','error':'something new'})=='needs_attention_unknown_failure'

def test_restart_budget_and_success():
    assert decision(restarts=2)=='needs_attention_restart_limit'
    assert decision(complete=True)=='complete'

def test_fingerprint_check_is_exact(tmp_path):
    p=tmp_path/'module.py';p.write_text('print(1)')
    frozen=tmp_path/'freeze_manifest.json';frozen.write_text('{}')
    protocol={'code':{'module.py':watch.sha(p)},'freeze_manifest_sha256':watch.sha(frozen)}
    assert watch.source_matches(protocol,tmp_path)==[]
    p.write_text('print(2)')
    assert watch.source_matches(protocol,tmp_path)==['module.py']
