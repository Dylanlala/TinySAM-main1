try:
    v10Detect
except NameError:
    from ultralytics.nn.modules.head import Detect
    v10Detect = Detect 