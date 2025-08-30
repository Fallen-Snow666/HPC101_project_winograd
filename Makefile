CC = nvcc
CFLAGS = -O3 -std=c++17
LFLAGS = -lcublas
TARGET = winograd
SOURCES = main.cu naive_conv.cu winograd_conv.cu

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
