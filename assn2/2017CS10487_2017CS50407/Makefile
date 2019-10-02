main:
	g++ main.cpp -lopencv_core -lopencv_imgcodecs `pkg-config opencv --cflags --libs`
	./a.out
main_knn:
	g++ main_knn_descriptor.cpp -lopencv_core -lopencv_imgcodecs `pkg-config opencv --cflags --libs`
	./a.out

clean:
	rm *.jpg
