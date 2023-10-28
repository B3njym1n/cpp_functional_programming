### CPP Functional Programming Notes

---

##### Books

- [Functional Programming in C++](https://www.manning.com/books/functional-programming-in-c-plus-plus), **Ivan Čukić**, Manning Publications, 2018

- Template Metaprogramming with C++, Marius Bancila

- C++ Template: The Complete Guide, 2nd edition, David Vandevoorde & Nicolai M. Josuttis & Douglas Gregor

##### Basics

- Function Composition

  ```C++
  #include <utility>

  template<typename F, typename G>
  struct Compose {
  	F _f;
  	G _g;

  	template<typename... Args>
  	auto operator()(Args... args) -> decltype(_f(_g(std::declval<Args>()...))) {
  		return _f(_g(std::forward<Args>(args)...));
  	}

  	template<typename F, typename G>
  	constexpr Compose(F&& f, G&& g) : _f(std::forward<F>(f)), _g(std::forward<G>(g)) {}

  };

  template<typename F, typename G>
  inline Compose<F, G> compose(F f, G g) {
  	return Compose<F, G>(f, g);
  }
  ```

- Currying

  ```C++
  #include <utility> // std::forward
  #include <tuple> // std::apply std::tuple_cat

  /**
  * can only apply to Callable object
  * In C++ callable object includes:
  *	pointer to function
  *	pointer to memberfunction
  *	functor
  *	lambda expression
  *	std::function is a template class that can hold any callable object
  */

  template<typename Function, typename... Args>
  struct curried {
  	using CapturedArgTuple = std::tuple<std::decay_t<Args>...>;
  	template<typename... Args>
  	static auto capture_by_copy(Args&&... args) {
  		return std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
  	}

  	template<typename... NewArgs>
  	auto operator() (NewArgs... args) const {
  		auto new_args = capture_by_copy(std::forward<NewArgs>(args)...);
  		auto all_args = std::tuple_cat(_captured, std::move(new_args));
  		if constexpr (std::is_invocable_v<Function, Args..., NewArgs...>)
  		{
  			return std::apply(_function, all_args);
  		}
  		else
  		{
  			return curried<Function, Args..., NewArgs...>(_function, all_args);
  		}
  	}


  	curried(Function func, Args... args):
  		_function(func)
  		, _captured(capture_by_copy(std::move(args)...)) {}

  	curried(Function func, std::tuple<Args... > args):
  		_function(func)
  		, _captured(std::move(args)) {}

  	Function _function;
  	CapturedArgTuple _captured;
  };
  ```

- fmap [fmap in c++](https://yapb-soc.blogspot.com/2012/10/fmap-in-c.html)

  The problem was that we wanted one *fmap* that worked on all STL-like containers, and one that worked on all smart pointers, but the two functions had the same signature and it wouldn't compile.

  ###### Tag Dispatch

  Before *std::enable_if*, there was [tag dispatch](http://www.generic-programming.org/languages/cpp/techniques.php). The idea was you start with your tags.

  ```c++
  struct sequence_tag {};
  struct pointer_tag {};
  struct other_tag {};

  // And then you define a traits class that defines the category of that type as a tag.

  template< class X > struct fmap_traits {
    typedef other_tag category;
  };

  template< class X > struct fmap_traits< std::vector<X> > {
    typedef sequence_tag category;
  };

  template< class X > struct fmap_traits< std::unique_ptr<X> > {
    typedef pointer_tag category;
  };

  // Now, rather than specializing fmap, we specialize fmap_impl which takes an extra argument, the category
  template< class F, template<class...>class S, class X,
    	   class R = typename std::result_of<F(X)>::type >
  S<R> fmap_impl( F&& f, const S<X>& s, sequence_tag ) {
      S<R> r;
      std::transform (
    	  std::begin(s), std::end(s),
    	  std::back_inserter(r),
    	  std::forward<F>(f)
      );
      return r;
  }

  template< class F, template<class...>class Ptr, class X,
      class R = typename std::result_of<F(X)>::type >
  Ptr<R> fmap_impl( F&& f, const Ptr<X>& p, pointer_tag )
  {
      return p != nullptr
    	  ? Ptr<R>( new R( std::forward<F>(f)(*p) ) )
    	  : nullptr;
  }

    // The job of fmap is now to dispatch to the appropriate fmap_impl.
    template< class F, class Functor,
    		class C = typename fmap_traits<Functor<X>>::category >
  auto fmap( F&& f, const Functor& fnct )
      -> decltype( fmap_impl( std::declval<F>(), fnct, C() ) );
  {
      return fmap_impl( std::forward<F>(f), fnct, C() );
  }

  ```

  This technique originally allowed STL algorithms to choose the most efficient implementation based on whether an iterator supported random access (it+n) or whether it allowed for assignment (it2\=it1) or not. **_The only problem is that we have to specialized fmap_traits for every single type on top of fmap_impl for each tag_**, though this is significantly less difficult than specializing fmap_impl for every type. Still, we can do better.

  ###### Type class dispatch

  First, instead of writing an *fmap_traits* class, we can use the *decltype* trick above to overload a function, *category*, that returns the correct tag, and just echos the type otherwise. We don't need to actually define it; a declaration will do.

```C++
      struct sequence_tag {};
      struct pointer_tag {};

      template< class X >
      X category( ... );

      template< class S >
      auto category( const S& s ) -> decltype( std::begin(s), sequence_tag() );

      template< class Ptr >
      auto category( const Ptr& p ) -> decltype( *p, p==nullptr, pointer_tag() );
```

Notice that if we supply *category* and *int*, it'll return an *int*, but if we give it a function pointer, it'll return *pointer_tag*! Why is that? Well, a function pointer is a pointer! You can dereference it and test it against null, so for this to work we have to add one extra layer of specialization.\\

```C++
      template< class T > struct Category {
          using type = decltype( category<T>(std::declval<T>()) );
      };

      template< class R, class ... X > struct Category< R(&)(X...) > {
          using type = R(&)(X...);
      };

      template< class T >
      using Cat = typename Category<T>::type;

      // category called on a function might return pointer_tag, but Cat<F>::type will be F.
      // Finally, instead of writing fmap_impl we will make a class called Functor that will implement fmap as a static member function. All we are doing is moving fmap_impl to Functor::fmap. fmap will then just call Functor::fmap.
      template< class... > struct Functor;

      template< class F, class FX, class Fun=Functor< Cat<FX> > >
      auto fmap( F&& f, FX&& fx )
          -> decltype( Fun::fmap( std::declval<F>(), std::declval<FX>() ) )
      {
          return Fun::fmap( std::forward<F>(f), std::forward<FX>(fx) );
      }

      // General case: composition
      template< class Function > struct Functor<Function> {
          template< class F, class G, class C = Composition<F,G> >
          static C fmap( F f, G g ) {
              C( std::move(f), std::move(g) );
          }
      };

      template<> struct Functor< sequence_tag > {
          template< class F, template<class...>class S, class X,
                    class R = typename std::result_of<F(X)>::type >
          static S<R> fmap( F&& f, const S<X>& s ) {
              S<R> r;
              std::transform( std::begin(s), std::end(s),
                              std::back_inserter(r),
                              std::forward<F>(f) );
              return r;
          }
      };

      template<> struct Functor< pointer_tag > {
          template< class F, template<class...>class Ptr, class X,
                    class R = typename std::result_of<F(X)>::type >
          static Ptr<R> fmap( F&& f, const Ptr<X>& p )
          {
              return p != nullptr
                  ? Ptr<R>( new R( std::forward<F>(f)(*p) ) )
                  : nullptr;
          }
      };

      struct UserDefined { /* ... */ };
      template<> struct Functor< UserDefined > {
          /* ... */
      };

      int main() {
          auto neg = [](int x){return -x;};
          std::unique_ptr<int> p( new int(5) );
          p = fmap( neg, fmap( neg, p ) );
          std::cout << "-5 = " << *p << std::endl;

          std::vector<int> w = { 1, 2, 3 };
          w = fmap( neg, w );
          std::copy( std::begin(w), std::end(w),
                     std::ostream_iterator<int>(std::cout," ") );
          std::cout << std::endl;
      }
```

_It is very important that **Functor\<T>::fmap** is static, or this will not work. One advantage is that we can still **further specialize fmap f**or different types. For example, we can't call our fmap on an std::array since it has no member function push_back(). Instead, we can specialize fmap for std::array inside Functor\<sequence>. A Functor specialization can overload as many or as few versions of fmap as it pleases._

At last, we not only have an fmap that works generically on STL containers, and all smart pointers, we have a technique that brings a different kind of polymorphism to C++. One that allows us to add specializations without modifying the previous ones. It's also surprisingly similar to the Haskell definition of *Functor*.

```c++
      class Functor f where
              fmap :: (a->b) -> f a -> f b
```

This is as if we had declared **_fmap_** like so:

```c++
      template< class F, template<class...>class Fnct, class X,
                class R = typename std::result_of<F(X)>,
                class Fun=Functor< Cat<Fnct<X>> > >
      Fnct<R> fmap( F&& f, const Fnct<X>& fx ) {
          return Fun::fmap( std::forward<F>(f), fx );
      }
```

- Maybe

- Monad IO

##### Functional Parser Implementation

- Parser Combinator [C++ 元编程之 Parser Combinator](https://netcan.github.io/2020/09/16/C-%E5%85%83%E7%BC%96%E7%A8%8B%E4%B9%8BParser-Combinator/) 借助 C++ 的 constexpr + lambda 能力，可以轻而易举的构造 Parser Combinator，实现一个 Parser 也没那么繁杂了

  > 没有类似 haskell parsec 的 monad parser gcc13 c++17 编译通过 msvc 好像不行

````C++
      #pragma once
      #include <type_traits> // std::is_same_v std::decay_t
      #include <utility> // std::forward ...
      #include <string>
      #include <string_view>
      #include <optional>


      namespace ParserCombinator {
          ///////////////////////////////////////////////////////////////////////////////
          // Parser a :: String -> Maybe (a, String)
          using ParserInput = std::string_view;
          template <typename T>
          using ParserResult = std::optional<std::pair<T, ParserInput>>;
          template <typename T>
          using Parser = auto(*)(ParserInput)->ParserResult<T>;

          template<typename>
          struct dump;

          // Partial template specialization
          // below ParserTrait is primary template
          template<typename T>
          class ParserTrait {
              using TDecay = std::decay_t<T>; // performs the type conversions equivalent to the ones performed when passing function arguments by value.
          public:
              using type = typename ParserTrait<decltype(&TDecay::operator())>::type;
          };

          template<typename T>
          using Parser_t = typename ParserTrait<T>::type;

          template<typename T, typename U>
          struct ParserTrait<auto (T::*)(ParserInput)->ParserResult<U>>
          {
              using type = U;
          };

          template<typename T, typename U>
          struct ParserTrait<auto (T::*)(ParserInput) const->ParserResult<U>> :
              ParserTrait<auto (T::*)(ParserInput)->ParserResult<U>> {};

          template<typename T>
          class CoercionTrait {
              using TDecay = std::decay_t<T>;
          public:
              using InputType = typename CoercionTrait<decltype(&TDecay::operator())>::InputType;
              using ResultType = typename CoercionTrait<decltype(&TDecay::operator())>::ResultType;
          };

          template<typename U, typename V>
          struct CoercionTrait<auto(*)(U)->V> {
              using InputType = U;
              using ResultType = V;
          };

          template<typename T, typename U, typename V>
          struct CoercionTrait<auto(T::*)(U)->V> {
              using InputType = U;
              using ResultType = V;
          };

          template<typename T, typename U, typename V>
          struct CoercionTrait<auto(T::*)(U) const->V> :
              CoercionTrait<auto(T::*)(U)->V> {};

          ///////////////////////////////////////////////////////////////////////////////
          constexpr auto makeCharParser(char c) {
              // CharParser :: Parser Char
              return [=](ParserInput s) -> ParserResult<char> {
                  if (s.empty() || c != s[0]) return std::nullopt;
                  return std::make_pair(s[0], ParserInput(s.data() + 1, s.size() - 1));
                  };
          };

          constexpr auto makeStringParser(std::string_view str) {
              // StringParser :: Parser String
              return [=](ParserInput s) -> ParserResult<std::string_view> {
                  if (s.empty() || s.find(str) != 0) return std::nullopt;
                  return std::make_pair(str, ParserInput(s.data() + str.size(), s.size() - str.size()));
                  };
          }

          constexpr auto oneOf(std::string_view chars) {
              // OneOf :: Parser Char
              return [=](ParserInput s) -> ParserResult<char> {
                  if (s.empty() || chars.find(s[0]) == std::string::npos) return std::nullopt;
                  return std::make_pair(s[0], ParserInput(s.data() + 1, s.size() - 1));
                  };
          }

          constexpr auto noneOf(std::string_view chars) {
              // NoneOf :: Parser Char
              return [=](ParserInput s) -> ParserResult<char> {
                  if (s.empty() || chars.find(s[0]) != std::string::npos) return std::nullopt;
                  return std::make_pair(s[0], ParserInput(s.data() + 1, s.size() - 1));
                  };
          }

          // fmap :: (a -> b) -> Parser a -> Parser b
          template<typename F, typename P,
              typename R = typename CoercionTrait<F>::ResultType>
          constexpr auto fmap(F&& f, P&& p) {
              static_assert(std::is_same_v<
                  typename CoercionTrait<F>::InputType,
                  Parser_t<P> >, "type mismatch!");
              return [=](ParserInput s) -> ParserResult<R> {
                  auto r = p(s);
                  if (!r) return std::nullopt;
                  return std::make_pair(f(r->first), r->second);
                  };
          }

          // bind :: Parser a -> (a -> Parser b) -> Parser b
          template<typename P, typename F>
          constexpr auto bind(P&& p, F&& f) {
              using R = std::invoke_result_t<F, Parser_t<P>, ParserInput>;
              return [=](ParserInput s) -> R {
                  auto r = p(s);
                  if (!r) return std::nullopt;
                  return f(r->first, r->second);
                  };
          }

          // operator| :: Parser a -> Parser a -> Parser a
          template<typename P1, typename P2>
          constexpr auto operator|(P1&& p1, P2&& p2) {
              return [=](ParserInput s) {
                  auto r1 = p1(s);
                  if (r1) return r1;
                  auto r2 = p2(s);
                  return r2;
                  };
          }

          // combine :: Parser a -> Parser b -> (a -> b -> c) -> parser c
          template<typename P1, typename P2, typename F,
              typename R = std::invoke_result_t<F, Parser_t<P1>, Parser_t<P2>>>
          constexpr auto combine(P1&& p1, P2&& p2, F&& f) {
              return [=](ParserInput s) -> ParserResult<R> {
                  auto r1 = p1(s);
                  if (!r1) return std::nullopt;
                  auto r2 = p2(r1->second);
                  if (!r2) return std::nullopt;
                  return std::make_pair(f(r1->first, r2->first), r2->second);
                  };
          };

          // operator> :: Parser a -> Parser b -> Parser a
          template<typename P1, typename P2>
          constexpr auto operator>(P1&& p1, P2&& p2) {
              return combine(std::forward<P1>(p1),
                  std::forward<P2>(p2),
                  [](auto&& l, auto) { return l; });
          };

          // operator< :: Parser a -> Parser b -> Parser b
          template<typename P1, typename P2>
          constexpr auto operator<(P1&& p1, P2&& p2) {
              return combine(std::forward<P1>(p1),
                  std::forward<P2>(p2),
                  [](auto, auto&& r) { return r; });
          };

          namespace detail {
              // foldL :: Parser a -> b -> (b -> a -> b) -> Parser b
              template<typename P, typename R, typename F>
              constexpr auto foldL(P&& p, R acc, F&& f, ParserInput in) -> ParserResult<R> {
                  while (true) {
                      auto r = p(in);
                      if (!r) return std::make_pair(acc, in);
                      acc = f(acc, r->first);
                      in = r->second;
                  }
              };
          };

          // many :: Parser a -> b -> (b -> a -> b) -> Parser b
          template<typename P, typename R, typename F>
          constexpr auto many(P&& p, R&& init, F&& f) {
              static_assert(std::is_same_v<std::invoke_result_t<F, R, Parser_t<P>>, R>,
                  "type mismatch!");
              return [p = std::forward<P>(p),
                  f = std::forward<F>(f),
                  init = std::forward<R>(init)](ParserInput s) -> ParserResult<R> {
                  return detail::foldL(p, init, f, s);
                  };
          };

          // atLeast :: Parser a -> b -> (b -> a -> b) -> Parser b
          template<typename P, typename R, typename F>
          constexpr auto atLeast(P&& p, R&& init, F&& f) {
              static_assert(std::is_same_v<std::invoke_result_t<F, R, Parser_t<P>>, R>,
                  "type mismatch!");
              return [p = std::forward<P>(p),
                  f = std::forward<F>(f),
                  init = std::forward<R>(init)](ParserInput s) -> ParserResult<R> {
                  auto r = p(s);
                  if (!r) return std::nullopt;
                  return detail::foldL(p, f(init, r->first), f, r->second);
                  };
          };

          // separatedBy :: Parser a -> Parser x -> b -> (b -> a -> b) -> Parser b
          template<typename P, typename X, typename R, typename F>
          constexpr auto separatedBy(P&& p, X&& x, R&& init, F&& f) {
              static_assert(std::is_same_v<std::invoke_result_t<F, R, Parser_t<P>>, R>,
                  "type mismatch!");
              return [p = std::forward<P>(p),
                  x = std::forward<X>(x),
                  f = std::forward<F>(f),
                  init = std::forward<R>(init)](ParserInput s) -> ParserResult<R> {
                  return detail::foldL(p > x, init, f, s);
                  };
          };

        template<typename P, typename R = Parser_t<P>>
        constexpr auto option(P&& p, R&& defaultV) {
          return [=](ParserInput s) -> ParserResult<R> {
            auto r = p(s);
            if (!r) return make_pair(defaultV, s);
            return r;
          };
        };
      };

       // test code
        using namespace ParserCombinator;
        auto pa = makeCharParser('a');
        auto pb = makeCharParser('b');
        auto pc = pa | pb;
        auto res = pc("bac");
        std::cout << res->first << " " << res->second << '\n';
        constexpr auto sign = option(oneOf("+-"), '+');
        constexpr auto number = atLeast(oneOf("1234567890"), 0l, [](long acc, char c) -> long {
          return acc * 10 + (c - '0');
        });
        constexpr auto integer = option(number, 0l);
        constexpr auto point = makeCharParser('.');
        constexpr auto decimal = point < option(number, 0l);
        constexpr auto value = combine(integer, decimal,
      				 [](long integer, long decimal) -> double {
      				   double d = 0.0;
      				   while (decimal) {
      				     d = (d + (decimal % 10)) * 0.1;
      				     decimal /= 10;
      				   }
      				   return integer + d;
      				 });

        constexpr auto floating = combine(sign, value, [](char sign, double d) -> double { return sign == '+' ? d : -d;});
        auto res2 = floating("-0.456");
        std::cout << res2->first << "\n";

	```
- [Monadic Parsing in C++](https://yapb-soc.blogspot.com/2012/11/monadic-parsing-in-c.html)

  > Bartosz Milewski's blog it's very detailed

##### Other references

[cpp_functional_programming](https://github.com/graninas/cpp_functional_programming/blob/master/README.md),graninas\\
````
