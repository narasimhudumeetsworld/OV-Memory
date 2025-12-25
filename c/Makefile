# =====================================================================
# OV-Memory: Makefile
# =====================================================================
# Build configuration for Fractal Honeycomb Graph Database
# License: Apache License 2.0
# =====================================================================

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -pthread -lm
LDFLAGS = -pthread -lm
LDLIBS = -lm

TARGET_LIB = libov_memory.so
TARGET_TEST = test_ov_memory
SOURCES = ov_memory.c
TEST_SOURCES = test_ov_memory.c ov_memory.c
HEADERS = ov_memory.h
OBJECTS = $(SOURCES:.c=.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

# Default target
all: $(TARGET_LIB) $(TARGET_TEST)

# Build shared library
$(TARGET_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	@echo "✅ Built shared library: $(TARGET_LIB)"

# Build test executable
$(TARGET_TEST): $(TEST_OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo "✅ Built test executable: $(TARGET_TEST)"

# Compile object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "✅ Compiled: $@"

# Run tests
test: $(TARGET_TEST)
	@echo "🧭 Running tests..."
	./$(TARGET_TEST)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET_LIB) $(TARGET_TEST)
	@echo "🗑️  Cleaned build artifacts"

# Install library (optional)
install: $(TARGET_LIB)
	mkdir -p /usr/local/lib
	mkdir -p /usr/local/include
	cp $(TARGET_LIB) /usr/local/lib/
	cp $(HEADERS) /usr/local/include/
	@echo "✅ Installed library and headers"

# Help target
help:
	@echo "Available targets:"
	@echo "  make all       - Build shared library and tests"
	@echo "  make test      - Run unit tests"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make install   - Install library system-wide"
	@echo "  make help      - Show this help message"

.PHONY: all test clean install help
