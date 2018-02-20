package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="sense")
public class Sense {
	private Long sense_id;
	private String sense_value;
	
	
	public Sense() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Sense(Long sense_id, String sense_value) {
		super();
		this.sense_id = sense_id;
		this.sense_value = sense_value;
	}
	
	@Id
	@Column(name = "sense_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getSense_id() {
		return sense_id;
	}

	public void setSense_id(Long sense_id) {
		this.sense_id = sense_id;
	}

	public String getSense_value() {
		return sense_value;
	}

	public void setSense_value(String sense_value) {
		this.sense_value = sense_value;
	}

	@Override
	public String toString() {
		return "Sense [sense_id=" + sense_id + ", sense_value=" + sense_value + "]";
	}
	
	
}
