from src.verification.environment import nano_problems, pinned_packages


def target_environment():
    return dict(machine='aarch64', python='3.6.9', ubuntu='18.04',
        hardware='NVIDIA Jetson Nano Developer Kit', l4t='# R32 (release), REVISION: 7.1,',
        packages=dict(pinned_packages(), torch='1.10.0+git36449ea', mediapipe='0.8.5',
                      opencv='4.1.1', matplotlib='2.1.1'),
        import_errors={}, cuda='10.2', cuda_available=True,
        tools=dict(ffmpeg=True, ffprobe=True, libx264=True), camera_accessible=True)


def test_documented_nano_profile_is_accepted_by_version_validator():
    assert nano_problems(target_environment()) == []


def test_conflicting_packages_and_wrong_platform_are_explicit_failures():
    environment = target_environment()
    environment['packages']['numpy'] = '1.24.0'
    environment['packages']['torch'] = '2.1.0'
    environment['packages']['mediapipe'] = '0.10.0'
    environment['machine'] = 'x86_64'
    environment['cuda_available'] = False
    problems = '\n'.join(nano_problems(environment))
    for name in ('numpy', 'torch', 'mediapipe', 'aarch64', 'CUDA'):
        assert name in problems


def test_missing_import_encoder_or_camera_cannot_pass_startup():
    environment = target_environment()
    environment['import_errors']['mediapipe'] = 'missing shared library'
    environment['tools']['libx264'] = False
    environment['camera_accessible'] = False
    problems = '\n'.join(nano_problems(environment))
    assert 'missing shared library' in problems
    assert 'libx264' in problems
    assert 'camera' in problems.lower()
    assert not any('camera' in p.lower() for p in nano_problems(environment, check_camera=False))
