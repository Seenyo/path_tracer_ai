Build Command
```bash
mkdir build
cd build
cmake ../CMakeLists.txt -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Complile Command
```bash
g++ -std=c++17 -fopenmp -O3 src/main.cpp src/renderer.cpp src/scene.cpp -I include -I external -I external/glm -o build/path_tracer_cpu
```

Run the exe
```bash
./build/path_tracer_cpu
```