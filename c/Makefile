.PHONY: all clean test build run

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -lm
TARGET = ov_memory
SOURCE = ov_memory.c

all: build

build:
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)
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
