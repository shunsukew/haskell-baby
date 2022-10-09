doubleMe x = x + x

doubleUs x y = doubleMe x + doubleMe y

doubleSmallNumber x = if x > 100
    then x
    else x * 2

conanO'Brien = "It's a-me, Conan O'Brien!"

length' xs = sum [1 | _ <- xs]

removeNonUpperCase xs = [c | c <- xs, c `elem` ['A'..'Z']]

addThree :: Int -> Int -> Int -> Int
addThree x y z = x + y + z

sayMe :: (Integral a) => a -> String
sayMe 1 = "One!"
sayMe 2 = "Two!"
sayMe 3 = "Three!"
sayMe x = "Not between 1 and 5"

factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)

addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

tell :: (Show a) => [a] -> String
tell [] = "The list is empty"
tell (x:[]) = "The list has one element: " ++ show x
tell (x:y:[]) = "The list has two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "The list is ling, The first two elements are: " ++ show x ++ " and " ++ show y

length'' :: (Num b) => [a] -> b
length'' [] = 0
length'' (_:xs) = 1 + length' xs

sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs

capital :: String -> String
capital "" = "Empty string"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]

bmiTell :: (RealFloat a) => a -> String
bmiTell bmi
    | bmi <= 18.5 = "You're underweight"
    | bmi <= 25.0 = "You're supposedly normal"
    | bmi <= 30.0 = "Let's exercise!"
    | otherwise   = "Congratulations!"

max' :: (Ord a) => a -> a -> a
max' a b
    | a > b = a
    | otherwise = b

myCompare :: (Ord a) => a -> a -> Ordering
a `myCompare` b
    | a > b = GT
    | a == b = EQ
    | otherwise = LT

initials :: String -> String -> String
initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
    where (f:_) = firstname
          (l:_) = lastname

calcBmi :: (RealFloat a) => [(a, a)] -> [a]
calcBmi xs = [bmi w h | (w, h) <- xs]
    where bmi weight height = weight / height ^ 2

cylinder :: (RealFloat a) => a -> a -> a
cylinder r h =
    let sideArea = 2 * pi * r * h
        topArea = pi * r ^2
    in sideArea + 2 * topArea

calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2]

head' :: [a] -> a
head' [] = error "No head for empty lists!"
head' (x:_) = x

head'' :: [a] -> a
head'' xs = case xs of [] -> error "No head for empty lists!"
                       (x:_) -> x

describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of [] -> "empty."
                                               [x] -> "a singleton list."
                                               xs -> "a longer list."

-- Recursion
maximum' :: (Ord a) => [a] -> a
maximum' [] = error "maximum of empty list"
maximum' [x] = x
maximum' (x:xs)
    | x > maxTail = x
    | otherwise = maxTail
    where maxTail = maximum' xs

replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
    | n <= 0 = []
    | otherwise = x:replicate' (n-1) x

take' :: (Num i, Ord i) => i -> [i] -> [i]
take' n _
    | n <= 0 = []
take' _ [] = []
take' n (x:xs) = x : take' (n-1) xs

reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse' xs ++ [x]

zip' :: [a] -> [b] -> [(a,b)]
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x, y):zip' xs ys

quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
    let smallerSorted = quicksort [a | a <- xs, a <= x]
        biggerSorted = quicksort [a | a <- xs, a > x]
    in smallerSorted ++ [x] ++ biggerSorted

-- Higher order functions
multThree :: (Num a) => a -> a -> a -> a  
multThree x y z = x * y * z

applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)

zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

flip' :: (a -> b -> c) -> (b -> a -> c)
flip' f = g
    where g x y = f y x

map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs)
    | p x = x : filter' p xs
    | otherwise = filter' p xs

largestDivisible :: (Integral a) => a
largestDivisible = head (filter p [100000, 99999..])
    where p x = x `mod` 3829 == 0

chain :: (Integral a) => a -> [a]
chain 1 = [1]
chain n
    | even n = n:chain (n `div` 2)
    | odd n = n:chain (n * 3 + 1)

numLongChains :: Int
numLongChains = length (filter (\xs -> length xs > 15) (map chain [1..100]))

sum'' :: (Num a) => [a] -> a
sum'' xs = foldl (\acc x -> acc + x) 0 xs

elem' :: (Eq a) => a -> [a] -> Bool
elem' y ys = foldl (\acc x -> if x == y then True else acc) False ys

map'' :: (a -> b) -> [a] -> [b]
map'' f xs = foldr (\x acc -> f x : acc) [] xs

maximum'' :: (Ord a) => [a] -> a
maximum'' = foldr1 (\x acc -> if x > acc then x else acc)

reverse'' :: [a] -> [a]
reverse'' = foldl (\acc x -> x : acc) []

sqrtSums :: Int
sqrtSums = length (takeWhile (<1000) (scanl1 (+) (map sqrt [1..]))) + 1

oddSquareSum :: Integer
oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]

