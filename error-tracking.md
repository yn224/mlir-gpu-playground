# Error Tracking for GPU execution
<mark>NOTE</mark> For examples to work, they:
1. Require manual annotation of main function `func.func @main{ ... }`
2. Don't go through the pass `refback-munge-calling-conventions`

## General
Omit func-bufferize --> finalizing-bufferize won't work

* LLVM15 - seems like it doesn't support printing by us?
```
/opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: in function `main':
/home/yn224/linalg:264: undefined reference to `print_memref_f32'
clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
```

## Matmul example
* `linalg-bufferize`
  * Further lowering required
* `func-bufferize`
  * Error
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: /lib/../lib64/Scrt1.o: in function `_start':
    (.text+0x20): undefined reference to `main'
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```

  * Solution: Manually annotate func.func @main{ return }

* `arith-bufferize`
  * Error
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: /lib/../lib64/Scrt1.o: in function `_start':
    (.text+0x20): undefined reference to `main'
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```

  * Solution: Manually annotate func.func @main{ return }

* `finalizing-bufferize`
  * Error
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: /lib/../lib64/Scrt1.o: in function `_start':
    (.text+0x20): undefined reference to `main'
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```
    
  * Solution: Manually annotate func.func @main{ return }

* `refback-munge-calling-conventions`
  * Error
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: /lib/../lib64/Scrt1.o: in function `_start':
    (.text+0x20): undefined reference to `main'
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: in function `refbackend_consume_func_return_mrf32':
    /home/yn224/linalg:10: undefined reference to `_mlir_ciface_refbackend_consume_func_return_mrf32'
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: /home/yn224/linalg:10: undefined reference to `_mlir_ciface_refbackend_consume_func_return_mrf32'
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```

  * Solution (for `_start`): Manually annotate func.func @main{ return }
  * Solution (for `_mlir_ciface_refbackend_consume_func_return_mrf32`): Don't use this pass

* `refback-munge-memref-copy`
  * Error
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: in function `refbackend_consume_func_return_mrf32':
    /home/yn224/linalg:11: undefined reference to `_mlir_ciface_refbackend_consume_func_return_mrf32'
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: in function `forward':
    /home/yn224/linalg:228: undefined reference to `print_memref_f32'
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: in function `refbackend_consume_func_return_mrf32':
    /home/yn224/linalg:11: undefined reference to `_mlir_ciface_refbackend_consume_func_return_mrf32'
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```

## Resnet18 example
* Error:
  * ```
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: exout.o: relocation R_X86_64_32 against `.rodata' can not be used when making a PIE object; recompile with -fPIC
    /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/../../../../bin/ld: final link failed: nonrepresentable section on output
    clang-15: error: linker command failed with exit code 1 (use -v to see invocation)
    ```
* Solution: `-no-pie` option added in `clang++` command in `run.sh`.