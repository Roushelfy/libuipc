"""
cuda_mixed entry point for the revolute joint limit GUI demo.

Run:
    LD_LIBRARY_PATH=build/build_impl_path6/Release/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=build/build_impl_path6/python/src uv run --project python --python 3.13 python \
        python/examples/cuda_mixed_revolute_joint_limit_viewer.py
"""

from revolute_joint_limit_gui_demo import main


if __name__ == "__main__":
    main(default_backend="cuda_mixed")
