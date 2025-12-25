.PHONY: all clean test build run

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm
TARGET = ov_memory
SOURCE = ov_memory.c

all: build

build: clean
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "✅ C compilation successful"

test: build
	@./$(TARGET)
	@echo "✅ C tests passed"

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o core
	@echo "✅ Cleaned build artifacts"

benchmark: build
	@echo "Running C benchmarks..."
	@time ./$(TARGET)
