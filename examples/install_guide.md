```bash
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN/
mkdir build
cd build/
cmake ..  -DCMAKE_INSTALL_PREFIX:PATH=/home/st_liu/workspace/pkgs/onednn3 ..
make -j
ctest && cmake --build . --target install

# Link oneDNN
export DNNLROOT=/home/st_liu/workspace/pkgs/onednn3/
g++ -I ${DNNLROOT}/include -L ${DNNLROOT}/lib64 getting_started.cpp -ldnnl -o start.o

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/st_liu/workspace/pkgs/onednn3/lib64
./start.o
# Example passed on CPU.
```