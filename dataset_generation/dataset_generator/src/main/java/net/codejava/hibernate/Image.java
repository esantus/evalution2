package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="image")
public class Image {
	private Long image_id;
	private String image_value;
	
	
	public Image() {
		super();
		// TODO Auto-generated constructor stub
	}


	public Image(Long image_id, String image_value) {
		super();
		this.image_id = image_id;
		this.image_value = image_value;
	}

	@Id
	@Column(name = "image_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getImage_id() {
		return image_id;
	}


	public void setImage_id(Long image_id) {
		this.image_id = image_id;
	}


	public String getImage_value() {
		return image_value;
	}


	public void setImage_value(String image_value) {
		this.image_value = image_value;
	}


	@Override
	public String toString() {
		return "Image [image_id=" + image_id + ", image_value=" + image_value + "]";
	}
	
}
