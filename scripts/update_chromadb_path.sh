#!/bin/bash
# Update ChromaDB path from polymath_fresh to polymath_v2

OLD_PATH="polymath_fresh"
NEW_PATH="polymath_v2"
DIR="/home/user/work/polymax"

echo "Updating ChromaDB path in all Python files..."
echo "  Old: $OLD_PATH"
echo "  New: $NEW_PATH"

# Find and update
for f in $(grep -rl "$OLD_PATH" "$DIR" --include="*.py" 2>/dev/null); do
    sed -i "s/$OLD_PATH/$NEW_PATH/g" "$f"
    echo "  Updated: $f"
done

echo "Done!"
