# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -pipe -std=c11
LDFLAGS = -lhs -lpthread -lm

# Project name
TARGET = regex_matcher

# Source files
SRCS = regex_matcher.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Install dependencies (Hyperscan)
install-deps:
	sudo apt update
	sudo apt install -y libhyperscan-dev

# Clean build files
clean:
	rm -f $(TARGET) $(OBJS)

# Run with example parameters (adjust as needed)
run: $(TARGET)
	./$(TARGET) patterns.txt input.txt output.txt metrics.csv

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: $(TARGET)

# Profile build
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: $(TARGET)

# Install to system (optional)
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/

.PHONY: all clean install-deps run debug profile install