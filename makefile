# CFLAGS with -march=native REMOVED to ensure compatibility.
# -D_GNU_SOURCE is kept here, so it can be removed from the .c file.
CFLAGS = -Wall -Wextra -O3 -pipe -std=gnu11 -D_GNU_SOURCE

# Linker flags for Hyperscan, pthreads, and math library
LDFLAGS = -lhs -lpthread -lm

# Project name
TARGET = regex_matcher

# Source files
SRCS = regex_matcher.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Rule to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean build files
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
